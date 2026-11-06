"""UAVDT ablation #2: DDA + similarity-gated retrieval + old FE (CBR+SA).

Same as group #5 (DDA + sim-gated retrieval, no FE), but enables the old FE
block with spatial attention only (no channel attention).
"""
_base_ = './p345_similarity_gated_shared_memory_drfpn_res18_uavdt.py'

model = dict(
    neck=dict(
        use_spatial_attn=True,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_similarity_gated_old_fe_drfpn_res18_uavdt'
