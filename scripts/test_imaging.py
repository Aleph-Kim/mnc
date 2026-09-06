"""Run inside the app environment: docker compose exec -T web python < scripts/test_imaging.py"""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image
from app.imaging.pipeline import generate_design
from app.imaging.segment import segment_regions, merge_small_regions
from app.imaging.regionmerge import merge_by_edge_evidence
from app.imaging.lineart import detect_line_layer
from app.imaging.faces import finalize_faces
from app.imaging.quantize import _pick_palette, to_lab

class ImagingRegression(unittest.TestCase):
    def test_single_region_merge(self):
        m = np.zeros((50,50), np.int32)
        result, colors = merge_small_regions(m, np.array([0]), np.array([[30,50,70]],np.uint8))
        self.assertTrue(np.array_equal(m,result))
        self.assertEqual(len(colors),1)

    def test_palette_keeps_black_white_and_distinct_small_colour(self):
        rgb = np.array([[210,210,150],[200,205,140],[180,190,130],[40,170,180],
                        [20,150,185],[10,130,190],[235,230,215],[20,25,20],[230,140,115]],np.uint8)
        lab = to_lab(rgb)
        selected = _pick_palette(lab,np.array([200,120,100,180,100,70,8,8,90]),6)
        for idx in (6,7,8):
            self.assertLess(np.linalg.norm(selected-lab[idx],axis=1).min(),1)

    def test_gradient_merges_but_real_edge_survives(self):
        h,w=120,160
        rgb=np.zeros((h,w,3),np.uint8)
        rgb[:]=np.linspace([20,150,180],[35,180,170],h)[:,None,:]
        labels=np.zeros((h,w),np.int32);labels[h//2:]=1
        m,c=segment_regions(labels)
        _, kept,_=merge_by_edge_evidence(m,c,rgb)
        self.assertEqual(len(kept),1)
        rgb[:h//2]=(250,220,40);rgb[h//2:]=(20,30,150)
        _, kept,_=merge_by_edge_evidence(m,c,rgb)
        self.assertEqual(len(kept),2)

    def test_thin_false_band_not_protected_as_stroke(self):
        rgb = np.full((100, 120, 3), (40, 170, 180), np.uint8)
        labels = np.zeros((100, 120), np.int32)
        labels[48:52] = 1
        m, c = segment_regions(labels)
        _, kept, _ = merge_by_edge_evidence(m, c, rgb)
        self.assertEqual(len(kept), 1)
        rgb[48:52] = (240, 40, 40)
        _, kept, _ = merge_by_edge_evidence(m, c, rgb)
        self.assertEqual(len(kept), 3)

    def test_coloured_ring_is_ink_and_pupil_is_fill(self):
        rgb=np.full((240,240,3),(80,180,180),np.uint8)
        cv2.circle(rgb,(120,120),75,(25,110,145),10)
        cv2.circle(rgb,(120,120),12,(15,20,15),-1)
        line,_=detect_line_layer(rgb)
        ring=np.zeros((240,240),np.uint8);cv2.circle(ring,(120,120),75,1,5)
        self.assertGreater(line[ring>0].mean(),0.9)
        self.assertFalse(line[120,120])

    def test_faces_keep_hole_and_disconnected_regions(self):
        labels=np.zeros((120,120),np.int32)
        cv2.circle(labels,(60,60),40,1,-1);cv2.circle(labels,(60,60),15,0,-1)
        m,c=segment_regions(labels)
        m,c=finalize_faces(m,c,np.zeros(labels.shape,bool))
        self.assertEqual(len(c),3)
        self.assertNotEqual(m[0,0],m[60,60])
        self.assertEqual(c[m[0,0]],c[m[60,60]])
        self.assertTrue((m>=0).all())

    def test_end_to_end_flat_and_shape(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            rgb=np.full((240,320,3),(90,160,110),np.uint8)
            src=root/'input.png';Image.fromarray(rgb).save(src)
            result=generate_design(src,8,root/'flat',process_max_dim=600)
            self.assertEqual(len(result.regions),1)
            self.assertEqual(len(np.unique(np.array(Image.open(result.preview_image_path)).reshape(-1,3),axis=0)),1)
            rgb=np.zeros((240,320,3),np.uint8)
            rgb[:]=np.linspace([20,140,190],[40,180,170],240)[:,None,:]
            cv2.circle(rgb,(160,120),50,(220,40,40),-1)
            Image.fromarray(rgb).save(src)
            result=generate_design(src,6,root/'shape',process_max_dim=600,mode='photo')
            preview=np.array(Image.open(result.preview_image_path))
            self.assertGreater(int(preview[225,300,0]),180)
            self.assertLess(int(preview[225,300,1]),80)
            summary=json.loads((root/'shape/debug/summary.json').read_text())
            self.assertEqual(summary['numbers_external'],0)
            self.assertEqual(summary['numbers_skipped'],0)

unittest.main(verbosity=2)
