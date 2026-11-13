"""Compare one interrupted run with an uninterrupted run on the same cached inputs."""
import tempfile
from pathlib import Path
import torch
import train
from data import HERE, read, save


class SimulatedInterruption(Exception):
    pass


torch.set_num_threads(4)
config = {**read(HERE/'config.json'), 'epochs': 3}
write_checkpoint = train.atomic_save
with tempfile.TemporaryDirectory(dir=HERE/'cache') as folder:
    root = Path(folder)
    train.HERE = root/'uninterrupted'
    reference = train.fit('fusion','original',131,config,'cpu')
    train.HERE = root/'resumed'

    def interrupt_after_checkpoint(path, state):
        write_checkpoint(path,state)
        if path.name == 'last.pt' and state['epoch'] == 1:
            raise SimulatedInterruption()

    train.atomic_save = interrupt_after_checkpoint
    try:
        train.fit('fusion','original',131,config,'cpu')
    except SimulatedInterruption:
        pass
    else:
        raise RuntimeError('Interruption did not happen')
    finally:
        train.atomic_save = write_checkpoint
    resumed = train.fit('fusion','original',131,config,'cpu',resume=True)
    a,b = [torch.load(Path(r['checkpoint']).with_name('last.pt'),map_location='cpu',weights_only=True)
           for r in (reference,resumed)]
    assert a['history'] == b['history'] and a['best'] == b['best']
    assert all(torch.equal(value,b['model'][key]) for key,value in a['model'].items())
save(HERE/'runs/resume_verification.json',{'passed':True,'device':'cpu','simulated_stop_after_epoch':1,
    'verified_epochs':3,'bitwise_equal_model_parameters':True,'identical_training_validation_history':True})
