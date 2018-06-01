
./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc\
 --set EXP_DIR seed_rng1701 RNG_SEED 1701

Find log file at */home/deeplearning_3/Aven/py-faster-rcnn/experiments/logs*

./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_16 caltech \
--set EXP_DIR seed_rng1701 RNG_SEED 1701

time ./tools/train_net.py --gpu 0 \
  --solver models/caltech/VGG16/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb caltech_train_1x \
  --iters 80000 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml





Fixing bug:

1.[](https://github.com/rbgirshick/fast-rcnn/issues/31)
Check failed: registry.count(type) == 1 (0 vs. 1) Unknown layer type: Python
  uncommnet  WITH_PYTHON_LAYER := 1 in Makefile.config
  run commands:
  make clear
  make all
  make -j128 && make pycaffe

2.[](https://github.com/rbgirshick/fast-rcnn/issues/37)
File "./tools/train_net.py", line 112, in <module>
    max_iters=args.max_iters)
  File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 160, in train_net
    model_paths = sw.train_model(max_iters)
  File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 111, in train_model
    model_paths.append(self.snapshot())
  File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 73, in snapshot
    self.bbox_stds[:, np.newaxis])
ValueError: operands could not be broadcast together with shapes (84,4096) (8,1)

Solution:
rename at least the layers that has different shape, e.g., the layer 'bbox_pred' and 'cls_score'.
And them you could finetune a new one.

In that case you could simply change the
{layer 'data': param_str: "'num_classes': 2"
layer 'cls_score': "num_output: 2",
layer 'bbox_pred': "num_output: 8". }
and then start the finetuning.

3.[](https://github.com/rbgirshick/fast-rcnn/issues/2)
Check failed: error == cudaSuccess (2 vs. 0)  out of memory

For test and plot image:
Add this to the end of function vis_detections in demo.py
```python
output = "output-{}.png".format(image_name.split(".",1)[0])
  fig.savefig(output)
  plt.close(fig)
```
