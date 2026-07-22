"""Frozen pretrained BC teachers: detach the BC part by relabeling the dataset offline."""
from impls.teacher.loader import FrozenTeacher, load_teacher, resolve_ckpt

__all__ = ["FrozenTeacher", "load_teacher", "resolve_ckpt"]
