"""Regenerate the user's reference without overwriting an existing design."""
import json
from app.core import design_store
from app.tasks.design_tasks import generate_design_task
source = design_store.load('d37ba4cd-e14e-44cc-bf5f-46d2e0d37e5d')
if source is None:
    raise RuntimeError('Reference design is not available')
new = design_store.create(source.image_id, source.original_image_path,
                          source.color_count, source.mode, source.recommended_colors)
generate_design_task(new.id)
result = design_store.load(new.id)
print(json.dumps({'id': new.id, 'status': result.status,
                  'url': f'http://localhost:8000/designs/{new.id}/view'}))
