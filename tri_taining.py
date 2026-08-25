for RW in hybrid sage ae; do
  python 03_rl.py --train --compare --compare_n 4 --mdp additive --data glorys --fixed pirata \
    --reward $RW --rl_steps 20000 --n_policies 10 --compare_seeds 2 --sage_stride 1 \
    --n_min 2 --n_max 12 --w_terminal 10 --output_dir outputs/ablation_$RW \
    --ae_checkpoint outputs/ae_best.pt --sage_checkpoint outputs/sage_best.pt
done
