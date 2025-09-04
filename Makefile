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

# Source files for the library
LIB_SOURCES = $(wildcard $(SRCDIR)/*.c)
# Object files for the library
LIB_OBJECTS = $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(LIB_SOURCES))

# Test source files
TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)
# Test object files
TEST_OBJECTS = $(patsubst $(TESTDIR)/%.c, $(BUILDDIR)/%.o, $(TEST_SOURCES))

# Executable names
EXAMPLE_EXEC = $(BINDIR)/mfcc_example
STREAM_EXEC = $(BINDIR)/stream_processor
TEST_EXEC = $(BINDIR)/run_tests

# Targets
.PHONY: all clean test

all: $(EXAMPLE_EXEC) $(STREAM_EXEC)

# Rule for the WAV file example executable
$(EXAMPLE_EXEC): $(LIB_OBJECTS) $(BUILDDIR)/main.o
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Rule for the stream processor executable
$(STREAM_EXEC): $(LIB_OBJECTS) $(BUILDDIR)/stream_processor.o
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Rule for the test executable
test: $(TEST_EXEC)
	./$(TEST_EXEC)

$(TEST_EXEC): $(LIB_OBJECTS) $(TEST_OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Rule for compiling library source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for compiling test source files
$(BUILDDIR)/%.o: $(TESTDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rules for compiling main executables' source files (from root dir)
$(BUILDDIR)/main.o: main.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c main.c -o $@

$(BUILDDIR)/stream_processor.o: stream_processor.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c stream_processor.c -o $@

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
