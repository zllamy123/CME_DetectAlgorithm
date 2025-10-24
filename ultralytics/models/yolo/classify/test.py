from ultralytics.models.yolo.classify import ClassificationTrainer


args = dict(model="yolo11x-cls.pt", data="/app/eyedetect/datasets", epochs=3)
trainer = ClassificationTrainer(overrides=args)
trainer.train()