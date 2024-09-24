

## Real Experiement:

### Training

```
# LADE_DUN 3stage at the 2nd phase, using synthetic data ckpt as pretrained 1st phase model
python train_and_test.py --train_phase 2 --clip_grad --batch_size 1  --template lade_dun   --outf ./exp/lade_dun_2rd/ --pretrained_model_path path_to_syn_data_ckpt  --method lade_dun --stage 3 --body_share_params 1  --resume_pre True --gpu_id 0

```


### Testing

```
# LADE_DUN 3stage
python train_and_test.py --train_phase 2 --test_mode 1  --batch_size 1 --template lade_dun  --outf ./exp/lade_dun_2rd_test/ --method lade_dun --stage 3 --body_share_params 1 --resume_ckpt_path path_to_real_data_ckpt --gpu_id 0
```

