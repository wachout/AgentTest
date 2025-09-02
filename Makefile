CC = gcc
CFLAGS = -Iinclude -Wall -g -fopenmp
LDFLAGS = -lm -fopenmp

SRCS = src/main.c src/xgboost.c src/data.c src/cJSON.c test_runner.c
OBJS = $(SRCS:.c=.o)

TARGET = xgboost

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
