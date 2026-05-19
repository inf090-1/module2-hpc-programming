SUBDIRS = 1-parallel-for 2-synchronization 3-reduction 4-scheduling 5-data-dependencies 6-matmul-optimizations

CC = gcc
CFLAGS = -Wall -fopenmp

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ || true

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean || true; \
	done
