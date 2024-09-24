

## Simulation Experiement:

### Training

```
# LADE_DUN 3stage at the 1st phase
python train_and_test.py --train_stage 1  --clip_grad --batch_size 4 --template lade_dun --outf ./exp/lade_dun_3stg_1st/ --method  lade_dun --stage 3 --body_share_params 0 --gpu_id 1 

# LADE_DUN 3stage at the 2nd phase
python train_and_test.py --train_stage 2  --clip_grad --batch_size 4 --template lade_dun --outf ./exp/lade_dun_3stg_2nd/ --resume_ckpt_path path_to_1st_phase_ckpt --method  lade_dun --stage 3 --body_share_params 0 --gpu_id 1 

# LADE_DUN 5/9/10stage at the 1st phase
python train_and_test.py --train_stage 1  --clip_grad --batch_size 1 --template lade_dun --outf ./exp/lade_dun_10stg_2nd/ --method  lade_dun --stage 10 --body_share_params 1 --gpu_id 1 

# LADE_DUN 5/9/10stage at the 2nd phase
python train_and_test.py --train_stage 2  --clip_grad --batch_size 1 --template lade_dun --outf ./exp/lade_dun_10stg_2nd/ --method  lade_dun --stage 10 --body_share_params 1 --resume_ckpt_path path_to_1st_phase_ckpt --gpu_id 1 
```


### Testing


```
# LADE_DUN 3stage
python train_and_test.py --train_stage 2  --test_mode 1  --template lade_dun --outf ./exp/lade_dun_3stg_2nd/ --method  lade_dun --stage 3 --body_share_params 0 --resume_ckpt_path path_to_ckpt --gpu_id 1

# LADE_DUN 5/9/10stage
python train_and_test.py --train_stage 2  --test_mode 1  --template lade_dun --outf ./exp/lade_dun_3stg_2nd/ --method  lade_dun --stage 10 --body_share_params 1 --resume_ckpt_path path_to_ckpt --gpu_id 1 
```

