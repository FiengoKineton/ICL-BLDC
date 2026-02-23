Layer A robustness: {
    'mean_abs_output_change': 3.66579032925074e-05, 
    'max_abs_output_change': 4.483645534492098e-05
    }
Closed-loop gain proxy (t=1): 0.04414469003677368

----------------------------------------------
keep=[0, 1, 2, 3, 4] -> {'final_train_loss': 0.0, 'final_val_loss': 0.0, 'robust_mean_abs_output_change': 1.7792197832022792e-05, 'closed_loop_gain_proxy_t1': 0.00957399606704712}
keep=[0, 1, 2, 3] -> {'final_train_loss': 0.0, 'final_val_loss': 0.0, 'robust_mean_abs_output_change': 2.8068318533769344e-05, 'closed_loop_gain_proxy_t1': 0.001210719347000122}
keep=[2, 3, 4] -> {'final_train_loss': 0.0, 'final_val_loss': 0.0, 'robust_mean_abs_output_change': 3.619552644522628e-05, 'closed_loop_gain_proxy_t1': 0.004023313522338867}

----------------------------------------------
Sensitivity sweep results:
{'lr': 0.0003, 'weight_decay': 0.0, 'final_train_loss': 1.0024286955595016, 'final_val_loss': 1.0011501722037792, 'final_grad_norm': 0.48597607482224703}
{'lr': 0.0003, 'weight_decay': 0.0001, 'final_train_loss': 1.0018448159098625, 'final_val_loss': 1.0000085569918156, 'final_grad_norm': 0.584612675011158}
{'lr': 0.0003, 'weight_decay': 0.001, 'final_train_loss': 1.0010435208678246, 'final_val_loss': 1.0020106509327888, 'final_grad_norm': 0.45741934329271317}
{'lr': 0.001, 'weight_decay': 0.0, 'final_train_loss': 1.0024119168519974, 'final_val_loss': 1.0017618872225285, 'final_grad_norm': 0.43615663005039096}
{'lr': 0.001, 'weight_decay': 0.0001, 'final_train_loss': 1.0023434348404408, 'final_val_loss': 1.0022296644747257, 'final_grad_norm': 0.37037915643304586}
{'lr': 0.001, 'weight_decay': 0.001, 'final_train_loss': 1.0039269141852856, 'final_val_loss': 1.0035291947424412, 'final_grad_norm': 0.6076360112056136}
{'lr': 0.003, 'weight_decay': 0.0, 'final_train_loss': 1.0037524215877056, 'final_val_loss': 1.004079658538103, 'final_grad_norm': 0.5109461708925664}
{'lr': 0.003, 'weight_decay': 0.0001, 'final_train_loss': 1.0030087903141975, 'final_val_loss': 1.0020078122615814, 'final_grad_norm': 0.49833835754543543}
{'lr': 0.003, 'weight_decay': 0.001, 'final_train_loss': 1.0034668818116188, 'final_val_loss': 1.0047144331037998, 'final_grad_norm': 0.5577268511988223}
Algorithmic stability proxy: {'val_loss_mean': 1.0064150566856067, 'val_loss_std': 0.0032343609049063334}
