#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

PROJECTS = \
	DeviceInfo \
	VAdd \
	MatMul \
	Pi \
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
