# Import before monarch to pre-load torch DSOs as, in exploded wheel flows,
# our RPATHs won't correctly find them.
import torch  # isort:skip
