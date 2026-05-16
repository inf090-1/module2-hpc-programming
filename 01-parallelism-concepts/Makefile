CC = gcc
CFLAGS = -O2 -Wall -lm

SOURCES = $(wildcard exercises/*.c)
EXAMPLES = $(wildcard examples/*.c)

TARGETS = $(SOURCES:.c=)
EXAMPLE_TARGETS = $(EXAMPLES:.c=)

.PHONY: all clean examples solutions

all: $(TARGETS) $(EXAMPLE_TARGETS)

examples: $(EXAMPLE_TARGETS)

solutions:
	@echo "Solutions available in exercises/solution/"

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) $(EXAMPLE_TARGETS)