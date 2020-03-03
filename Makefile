# written by junying, 2020-03-03

# rar a -v40000k ssd_300_vgg.ckpt.rar ssd_300_vgg.ckpt.data-00000-of-00001 ssd_300_vgg.ckpt.index
# unrar x ssd_300_vgg.ckpt.part1.rar 

init:
	@cd checkpoints && unrar x ssd_300_vgg.ckpt.part1.rar

clean:
	@find . -name "*.pyc"|xargs rm