
"""
### Orientation note:


Dimension (same for LUMIR and OASIS):
    160/192/224
     LR/ SI/ AP
    For brain: LR is the shortest, AP is the longest

Orientation
    ITK-SNAP needs to be verified!!!

    LUMIR:
        In ITK-SNAP:
            RAS+ or (LPI-)
                i: L->R (160)
                j: P->A (224)
                k: I->S (192)
            affine:
                1, 0, 0, 0
                0, 1, 0, 0
                0, 0, 1, 0
                0, 0, 0, 1

    OASIS:
        In ITK-SNAP:
            LIA+ (or RSP-)
                i: R->L (160)
                j: S->I (192)
                k: P->A (224)
            affine:
                -1,  0,  0,   80
                 0,  0,  1, -112
                 0, -1,  0,   96
                 0,  0,  0,    1
"""
