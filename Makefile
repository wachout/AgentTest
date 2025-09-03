# Makefile for MFCC Implementation

# Compiler and flags
CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -O2
LDFLAGS = -lm

# Directories
SRCDIR = src
INCDIR = include
TESTDIR = tests
BUILDDIR = build
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
# Object files
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SOURCES))

# Test source files
TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)
# Test object files
TEST_OBJECTS = $(patsubst $(TESTDIR)/%.c, $(BUILDDIR)/%.o, $(TEST_SOURCES))

# Executable name
EXECUTABLE = $(BINDIR)/mfcc_example
TEST_EXECUTABLE = $(BINDIR)/run_tests

# Targets
.PHONY: all clean test

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(BUILDDIR)/main.o
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(OBJECTS) $(TEST_OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(TESTDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Special rule for main.c which is not in src
$(BUILDDIR)/main.o: main.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c main.c -o $@


clean:
	rm -rf $(BUILDDIR) $(BINDIR)

list:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Test Sources: $(TEST_SOURCES)"
	@echo "Test Objects: $(TEST_OBJECTS)"
