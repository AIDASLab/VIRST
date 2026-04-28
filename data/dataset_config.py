import os
import os.path as osp


_DATA_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.dirname(_DATA_DIR)
_DEFAULT_DATASET_ROOT = osp.join(_REPO_ROOT, "dataset")


def _dataset_root(*env_names: str, default_subdir: str | None = None) -> str:
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            return osp.abspath(osp.expanduser(env_value))

    if default_subdir is None:
        return osp.abspath(_DEFAULT_DATASET_ROOT)
    return osp.abspath(osp.join(_DEFAULT_DATASET_ROOT, default_subdir))


# LISA
LISA_ROOT = _dataset_root("VIRST_LISA_ROOT", default_subdir="LISA")

# ChatUniVi
ChatUniVi_ROOT = _dataset_root("VIRST_CHATUNIVI_ROOT", default_subdir="Chat-UniVi-Instruct")
MIMIC_imageonly = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly.json"),
    "CDG": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/CDG/images"),
    "LA": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/LA/images"),
    "SD": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/SD/images"),
}
VIDEO = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/video_chat.json"),
    "VIDEO": osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/Activity_Videos"),
}
SQA = {
    "chat_path": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/llava_train_QCM-LEA.json"),
    "ScienceQA": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/train"),
}

# RVOS
RVOS_ROOT = _dataset_root("VIRST_RVOS_ROOT", default_subdir="RVOS_ROOT")
RVOS_DATA_INFO = {
    "mevis_train": (osp.join(RVOS_ROOT, "mevis/train"), osp.join(RVOS_ROOT, "mevis/train/meta_expressions.json")),
    "mevis_valid": (osp.join(RVOS_ROOT, "mevis/valid_u"), osp.join(RVOS_ROOT, "mevis/valid_u/meta_expressions.json")),
    "mevis_test": (osp.join(RVOS_ROOT, "mevis/valid"), osp.join(RVOS_ROOT, "mevis/valid/meta_expressions.json")),
    "refytvos_train": (osp.join(RVOS_ROOT, "refytvos/train"), osp.join(RVOS_ROOT, "refytvos/meta_expressions/train/meta_expressions.json")),
    "refytvos_valid": (osp.join(RVOS_ROOT, "refytvos/valid"), osp.join(RVOS_ROOT, "refytvos/meta_expressions/valid/meta_expressions.json")),
    "davis17_train": (osp.join(RVOS_ROOT, "davis17/train"), osp.join(RVOS_ROOT, "davis17/meta_expressions/train/meta_expressions.json")),
    "davis17_valid": (osp.join(RVOS_ROOT, "davis17/valid"), osp.join(RVOS_ROOT, "davis17/meta_expressions/valid/meta_expressions.json")),
    "revos_train": (osp.join(RVOS_ROOT, "ReVOS"), osp.join(RVOS_ROOT, "ReVOS/meta_expressions_train_.json")),
    "revos_valid": (osp.join(RVOS_ROOT, "ReVOS"), osp.join(RVOS_ROOT, "ReVOS/meta_expressions_valid_.json")),
    "lvvis_train": (osp.join(RVOS_ROOT, "lvvis/train"), osp.join(RVOS_ROOT, "lvvis/train/meta_expressions.json")),
}

VQA_VIDEO_ROOT = _dataset_root("VIRST_VQA_VIDEO_ROOT", default_subdir="LLaVA-Video-178K/0_30_s_academic_v0_1")
