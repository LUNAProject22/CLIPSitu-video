from vidsitu_code.mdl_sf_base import (
    SFBase,
    VerbModel,
    LossB,
    Simple_GPT2_New,
    MLP_Simple_GPT2_New,
    LossLambda,
    Simple_TxDec,
    Simple_TxEncDec,
    SFPreFeats_TxDec,
    MLP_TxDec,
    TxEncDec,
    XTF_TxEncDec, XTF_TxEncDec_wObj
)

from vidsitu_code.evl_vsitu import (
    EvalB,
    EvalB_Gen,
    EvalB_Acc,
)
from vidsitu_code.mdl_evrel import (
    Simple_EvRel_Roberta,
    SFPret_SimpleEvRel,
    SFPret_OnlyVb_SimpleEvRel,
    SFPret_OnlyVid_SimpleEvRel,
    Simple_TxEncEvRel,
)


def get_mdl_loss_eval(cfg):
    assert cfg.task_type in set(["vb", "vb_arg", "evrel", "evforecast"])
    # if cfg.task_type == "vb":
    #     assert cfg.mdl.mdl_name == "sf_base"
    #     return {"mdl": SFBase, "loss": LossB, "evl": EvalB}
    if cfg.task_type == "vb":
        return {"mdl": VerbModel, "loss": LossB, "evl": EvalB}
    # elif cfg.task_type == "vb_arg":
    #     if cfg.mdl.mdl_name == "new_gpt2_only":
    #         return {"mdl": Simple_GPT2_New, "loss": LossLambda, "evl": EvalB_Gen}

    #     elif cfg.mdl.mdl_name == "tx_only":
    #         return {"mdl": Simple_TxDec, "loss": LossLambda, "evl": EvalB_Gen}
    #     elif cfg.mdl.mdl_name == "txed_only":
    #         return {"mdl": Simple_TxEncDec, "loss": LossLambda, "evl": EvalB_Gen}

    #     elif cfg.mdl.mdl_name == "sfpret_txed_vbarg":
    #         return {"mdl": SFPreFeats_TxDec, "loss": LossLambda, "evl": EvalB_Gen}
    #     elif cfg.mdl.mdl_name == "sfpret_txe_txd_vbarg":
    #         return {"mdl": SFPreFeats_TxEncDec, "loss": LossLambda, "evl": EvalB_Gen}
    
    elif cfg.task_type == "vb_arg":
        # if cfg.mdl.mdl_name == "mlp_gpt22":
        #     return {"mdl": MLP_Simple_GPT2_New, "loss": LossLambda, "evl": EvalB_Gen}
        # elif cfg.mdl.mdl_name == "mlp_txed_vbarg":   # simple MLP
        #     return {"mdl": MLP_TxDec, "loss": LossLambda, "evl": EvalB_Gen}
        # elif cfg.mdl.mdl_name == "txe_txd_vbarg":    # Transformer
        #     return {"mdl": TxEncDec, "loss": LossLambda, "evl": EvalB_Gen}
        # elif cfg.mdl.mdl_name == "xtf_txe_txd_vbarg": # XTF
        #     return {"mdl": XTF_TxEncDec, "loss": LossLambda, "evl": EvalB_Gen}
        if cfg.mdl.mdl_name == "xtf_txe_txd_vbarg_obj": # XTF
            return {"mdl": XTF_TxEncDec_wObj, "loss": LossLambda, "evl": EvalB_Gen}
            
    elif cfg.task_type == "evrel":
        if cfg.mdl.mdl_name == "rob_evrel":
            return {
                "mdl": Simple_EvRel_Roberta,
                "loss": LossLambda,
                "evl": EvalB_Acc,
            }
        elif cfg.mdl.mdl_name == "txe_evrel":
            return {"mdl": Simple_TxEncEvRel, "loss": LossLambda, "evl": EvalB_Acc}
        elif cfg.mdl.mdl_name == "sfpret_evrel":
            return {"mdl": SFPret_SimpleEvRel, "loss": LossLambda, "evl": EvalB_Acc}
        elif cfg.mdl.mdl_name == "sfpret_vbonly_evrel":
            return {
                "mdl": SFPret_OnlyVb_SimpleEvRel,
                "loss": LossLambda,
                "evl": EvalB_Acc,
            }
        elif cfg.mdl.mdl_name == "sfpret_onlyvid_evrel":
            return {
                "mdl": SFPret_OnlyVid_SimpleEvRel,
                "loss": LossLambda,
                "evl": EvalB_Acc,
            }
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
