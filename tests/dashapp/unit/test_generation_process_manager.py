from scalebridge.dashapp.services.generation.execution import MANAGER
def test_process_manager_initial_snapshot_has_status():
 assert MANAGER.snapshot()['status'] in {'not_started','completed','failed','stopped'}
