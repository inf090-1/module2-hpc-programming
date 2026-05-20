SUBDIRS = 0-parallel-hello 1-parallel-for 2-reduction 3-synchronization 4-scheduling 5-matmul

CC = gcc
CFLAGS = -O3 -Wall -fopenmp

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ || true

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean || true; \
	done
