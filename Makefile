# Makefile for GBDT project

# Compiler
CC = gcc

# Compiler flags
# Add -lm for the math library (e.g., for exp() in gbdt.c)
CFLAGS = -Wall -Wextra -std=c99 -Isrc -O2
LDFLAGS = -lm

# Source directory
SRC_DIR = src

# Object directory
OBJ_DIR = obj

# Executable name
EXEC = gbdt_run

# Source files (now includes all .c files)
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Default target
all: $(EXEC)

# Create object directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link object files to create executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean up build artifacts
clean:
	rm -rf $(OBJ_DIR) $(EXEC) *.dat

.PHONY: all clean
