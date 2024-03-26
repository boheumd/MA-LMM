torchrun --nproc_per_node=4 \
    --master_port=34653 \
    train.py \
    --cfg-path lavis/projects/malmm/cls_lvu.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    model.num_frames 100 \
    datasets.lvu_cls.history 100 \
    datasets.lvu_cls.task relationship \
    datasets.lvu_cls.stride 20 \
    run.init_lr 1e-4 \
    run.max_epoch 20 \
    run.num_beams 5 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate False \
    run.report_metric True \
    run.prefix train 
    # run.resume_ckpt_path
