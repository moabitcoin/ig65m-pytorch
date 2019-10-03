dockerimage ?= moabitcoin/ig65m-pytorch
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker run                              \
	  --ipc=host                             \
	  -it                                    \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  --entrypoint=/bin/bash $(dockerimage)

r: run


gpu:
	@docker run                              \
	  --runtime=nvidia                       \
	  --ipc=host                             \
	  -it                                    \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  --entrypoint=/bin/bash $(dockerimage)

g: gpu


webcam:
	@docker run                              \
	  --ipc=host                             \
	  -it                                    \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  --device=/dev/video0                   \
	  --entrypoint=/bin/bash $(dockerimage)

w: webcam


publish:
	@docker image save $(dockerimage) \
	  | pv -N "Publish $(dockerimage) to $(sshopts)" -s $(shell docker image inspect $(dockerimage) --format "{{.Size}}") \
	  | ssh $(sshopts) "docker image load"

p: publish


.PHONY: install i run r update u webcam w gpu g publish p
