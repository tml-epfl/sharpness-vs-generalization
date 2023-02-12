### CNNs: avg_l2
for rho in 0.05 0.1 0.2 0.4; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_resnet_avg_l2 --model_path="${model_path}"
    done &
done
for rho in 0.05 0.1 0.2 0.4; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_resnet_avg_l2 --model_path="${model_path}"
    done &
done
for rho in 0.05 0.1 0.2 0.4; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_resnet_avg_l2 --model_path="${model_path}"
    done &
done

### CNNs: avg_linf
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_resnet_avg_linf --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_resnet_avg_linf --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_resnet_avg_linf --model_path="${model_path}"
    done &
done

### CNNs: max_l2
for rho in 0.25 0.5 1.0 2.0; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_resnet_max_l2 --model_path="${model_path}"
    done &
done
for rho in 0.25 0.5 1.0 2.0; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_resnet_max_l2 --model_path="${model_path}"
    done &
done
for rho in 0.25 0.5 1.0 2.0; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_resnet_max_l2 --model_path="${model_path}"
    done &
done

### CNNs: max_linf
for rho in 0.0005 0.001 0.002 0.004; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.001 0.002 0.004 0.008; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.001 0.002 0.004 0.008; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done





### ViTs: avg_l2
for rho in 0.005 0.01 0.02 0.04; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_vit_avg_l2 --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_vit_avg_l2 --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=avg_l2 --log_folder=logs_final_vit_avg_l2 --model_path="${model_path}"
    done &
done

### ViTs: avg_linf
for rho in 0.01 0.02 0.04 0.08; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_vit_avg_linf --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_vit_avg_linf --model_path="${model_path}"
    done &
done
for rho in 0.1 0.2 0.4 0.8; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=avg_linf --log_folder=logs_final_vit_avg_linf --model_path="${model_path}"
    done &
done

### ViTs: max_l2
for rho in 0.025 0.05 0.1 0.2; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_vit_max_l2 --model_path="${model_path}"
    done &
done
for rho in 0.5 1.0 2.0 4.0; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_vit_max_l2 --model_path="${model_path}"
    done &
done
for rho in 0.5 1.0 2.0 4.0; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_l2 --log_folder=logs_final_vit_max_l2 --model_path="${model_path}"
    done &
done

### ViTs: max_linf
for rho in 0.00001 0.00002 0.00004 0.00008; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.0005 0.001 0.002 0.004; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.0005 0.001 0.002 0.004; do
    for model_path in /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200*; do
        python eval_sharpness.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=128 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done



####### Ablation for different `m` for the best sharpness definition
for rho in 0.001 0.002 0.004 0.008; do
    ls /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=16 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.0005 0.001 0.002 0.004; do
    ls /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=16 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done

for rho in 0.001 0.002 0.004 0.008; do
    ls /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=32 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.0005 0.001 0.002 0.004; do
    ls /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=32 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done

for rho in 0.001 0.002 0.004 0.008; do
    ls /tmldata1/andriush/overfit/models/2023*model=resnet18*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=resnet18 --model_width=64 --n_eval_sharpness=1024 --bs_sharpness=64 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_resnet_max_linf --model_path="${model_path}"
    done &
done
for rho in 0.0005 0.001 0.002 0.004; do
    ls /tmldata1/andriush/overfit/models/2023*model=vit_exp*lr_schedule=cyclic*epoch=200* | shuf | while read model_path; do 
        python eval_sharpness.py --max_train_error=0.01 --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval_sharpness=1024 --bs_sharpness=64 --rho=$rho --adaptive --normalize_logits --n_iters=20 --algorithm=m_apgd_linf --log_folder=logs_final_vit_max_linf --model_path="${model_path}"
    done &
done

