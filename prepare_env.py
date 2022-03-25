import os
from transformers import T5Config


MODEL_DIR = "pretrained_models/t5_2l_8h_512d_2048ff"


def main():
    trainer_config = T5Config.from_pretrained("t5-large")

    student_config_dict = trainer_config.to_dict()  # makes it mutable
    student_config_dict["d_ff"] = 2048
    student_config_dict["d_model"] = 512
    student_config_dict["num_heads"] = 8
    student_config_dict["num_layers"] = 2
    student_config_dict["num_decoder_layers"] = 2

    student_config = T5Config.from_dict(student_config_dict)

    os.makedirs(MODEL_DIR, exist_ok=True)

    student_config.save_pretrained("tiny_model_config")

    # Distillation

    trainer_config = T5Config.from_pretrained("t5-small")

    student_config_dict = trainer_config.to_dict()  # makes it mutable
    student_config_dict["d_ff"] = 2048
    student_config_dict["d_model"] = 512
    student_config_dict["num_heads"] = 8
    student_config_dict["num_layers"] = 6
    student_config_dict["num_decoder_layers"] = 6

    student_config = T5Config.from_dict(student_config_dict)

    MODEL_DIR = "lfom_distilt5_6l_8h_512d_2048ff"
    os.makedirs(MODEL_DIR, exist_ok=True)

    student_config.save_pretrained("lfom_distilt5_6l_8h_512d_2048ff")


if __name__ == "__main__":
    main()
