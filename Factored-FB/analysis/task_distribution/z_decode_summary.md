# What each training-z encodes (held-out R^2 / AUC, mean±std over seeds)

| method | cube_xyz | cube_lift | hand_cube_dist | gripper_auc |
|---|---|---|---|---|
| fb_state_plain | 0.30±0.01 | 0.02±0.00 | 0.09±0.01 | 0.75±0.01 |
| psm_flowbc_state | 0.85±0.01 | 0.65±0.04 | 0.31±0.04 | 0.94±0.01 |
| psm_o1000 | 0.90±0.00 | 0.81±0.01 | 0.76±0.01 | 0.97±0.00 |
| fb_z128 | 0.57±0.01 | 0.30±0.03 | 0.19±0.01 | 0.91±0.01 |
| fb_state_iwr | 0.65±0.00 | 0.79±0.01 | 0.21±0.00 | 0.92±0.00 |
| fb_state_iwr_wz | 0.63±0.01 | 0.77±0.02 | 0.21±0.00 | 0.92±0.01 |