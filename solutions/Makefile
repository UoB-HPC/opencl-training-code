
PROJECTS = \
	Bilateral \
	HostDevTransfer \
	NBody \
	NBody-GL \
	NBody-GL-VBO


all:
	@for p in $(PROJECTS); do\
		$(MAKE) -C $$p; \
	done

clean:
	@for p in $(PROJECTS); do\
		$(MAKE) -C $$p clean; \
	done
