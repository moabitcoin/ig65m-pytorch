dockerimage ?= moabitcoin/ig65m-pytorch
dockerfile ?= Dockerfile
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker run -it --rm -v $(srcdir):/usr/src/app/  \
	                     -v $(datadir):/data         \
	                     --entrypoint=/bin/bash $(dockerimage)

r: run


webcam:
	@docker run -it --rm -v $(srcdir):/usr/src/app/  \
	                     -v $(datadir):/data         \
	                     --device=/dev/video0        \
	                     --entrypoint=/bin/bash $(dockerimage)

w: webcam


.PHONY: install i run r update u webcam w
