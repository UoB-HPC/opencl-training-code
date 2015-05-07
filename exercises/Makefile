
PROJECTS = \
	DeviceInfo \
	Bilateral \
	HostDevTransfer \
	NBody \
	NBody-GL


all:
	@for p in $(PROJECTS); do\
		$(MAKE) -C $$p; \
	done

clean:
	@for p in $(PROJECTS); do\
		$(MAKE) -C $$p clean; \
	done
