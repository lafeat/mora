
######################################adp   3
for model_old in './checkpoints/adp/seed_0/3_ResNet20/epoch_200.pth' ; do
  param=" --version=standard-t-pre  --model_file=$model_old
  --log_path=./evaluation_result/adp/seed_0/3_ResNet20/frize_0.01_standard-t-pre-early-stop.txt
  --scale --epsilon=0.01 --decay_step=cosine --batch_size=1000 --n_iter=100
  --n_ex=10000 --gpu=2   --float_dis=0.2 --model_name=our_adp_3_cosine_full --ensemble_pattern=softmax"
  # shellcheck disable=SC2016
  echo line='--------------------------------------------------------------------------'
  echo $param
  python eval_mora.py  $param
done

#####################################adp   3
for model_old in './checkpoints/adp/seed_0/3_ResNet20/epoch_200.pth' ; do
  param=" --version=standard-t-pre  --model_file=$model_old
  --log_path=./evaluation_result/adp/seed_0/3_ResNet20/frize_0.01_standard-t-pre-early-stop.txt
  --scale --epsilon=0.01 --decay_step=cosine --batch_size=1000 --n_iter=100
  --n_ex=10000 --gpu=2   --float_dis=0.8 --model_name=our_adp_3_cosine_full --ensemble_pattern=voting"
  # shellcheck disable=SC2016
  echo line='--------------------------------------------------------------------------'
  echo $param
  python eval_mora.py  $param
done

#######################################adp   3
for model_old in './checkpoints/adp/seed_0/3_ResNet20/epoch_200.pth' ; do
  param=" --version=standard-t-pre  --model_file=$model_old
  --log_path=./evaluation_result/adp/seed_0/3_ResNet20/frize_0.01_standard-t-pre-early-stop.txt
  --scale --epsilon=0.01 --decay_step=cosine --batch_size=1000 --n_iter=100
  --n_ex=10000 --gpu=2   --float_dis=0.0 --model_name=our_adp_3_cosine_full --ensemble_pattern=logits"
  # shellcheck disable=SC2016
  echo line='--------------------------------------------------------------------------'
  echo $param
  python eval_mora.py  $param
done
