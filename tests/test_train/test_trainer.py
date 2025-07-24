import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from types import SimpleNamespace
import miniformer.train.trainer as trainer_mod


def test_main_default(monkeypatch):
    # prepare dummy config
    cfg = SimpleNamespace(
        seed=123,
        task="classification",
        test_path="",
        work_dir="wd",
        experiment_name="exp",
        logger="tensorboard",
        early_stopping_patience=1,
        checkpoint_metric="val_loss",
        gpus=0,
        max_epochs=1,
        precision=32,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5
    )
    # patch from_cli
    monkeypatch.setattr(trainer_mod.TrainConfig, "from_cli", staticmethod(lambda: cfg))

    calls = []
    # patch seed_everything
    monkeypatch.setattr(trainer_mod.pl, "seed_everything", lambda *args, **kwargs: calls.append(("seed", args, kwargs)))
    # dummy DataModule and Model
    dummy_dm = object()
    dummy_model = object()
    monkeypatch.setattr(trainer_mod, "MiniFormerDataModule", lambda cfg, tok: dummy_dm)
    monkeypatch.setattr(trainer_mod, "MiniFormerLitModule", lambda cfg: dummy_model)
    # patch loggers - need to match the actual function signatures
    class DummyLogger:

        def __init__(self, *args, **kwargs):
            self.log_dir = "/fake/log/dir/version_0"

        def __str__(self):
            return "tb_logger"

    def create_logger(*args, **kwargs):
        return DummyLogger()
    
    monkeypatch.setattr(trainer_mod, "TensorBoardLogger", create_logger)
    monkeypatch.setattr(trainer_mod, "WandbLogger", create_logger)
    monkeypatch.setattr(trainer_mod, "CSVLogger", create_logger)
    # patch callbacks

    def create_callback(**kwargs):
        return "ckpt_cb"
    
    monkeypatch.setattr(trainer_mod, "ModelCheckpoint", create_callback)
    monkeypatch.setattr(trainer_mod, "EarlyStopping", lambda **kwargs: "es_cb")
    # dummy Trainer

    class DummyTrainer:

        def __init__(self, **kwargs):
            calls.append(("trainer_init", kwargs))

        def fit(self, model, datamodule):
            calls.append(("fit", model, datamodule))

        def validate(self, model, datamodule, verbose=False):
            calls.append(("validate", model, datamodule, verbose))

        def test(self, datamodule):
            calls.append(("test", datamodule))
    monkeypatch.setattr(trainer_mod.pl, "Trainer", DummyTrainer)

    # run main
    trainer_mod.main()

    # assertions
    assert ("seed", (cfg.seed,), {"workers": True}) in calls
    # Trainer instantiation
    trainer_calls = [c for c in calls if c[0] == "trainer_init"]
    assert trainer_calls, "Trainer was not instantiated"
    init_kwargs = trainer_calls[0][1]
    assert init_kwargs["max_epochs"] == cfg.max_epochs
    assert init_kwargs["accelerator"] == "cpu"
    assert init_kwargs["devices"] == 1
    assert str(init_kwargs["logger"]) == "tb_logger"
    # fit call
    assert ("fit", dummy_model, dummy_dm) in calls
    # validate call (always happens, even with empty cfg.test_path)
    assert ("validate", dummy_model, dummy_dm, False) in calls
    # no test call
    assert all(c[0] != "test" for c in calls)
