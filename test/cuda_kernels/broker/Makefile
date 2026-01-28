# Makefile for Broker Queue tests

NVCC = nvcc
NVCC_FLAGS = -std=c++14 -arch=sm_80 -O3 -lineinfo
INCLUDES = -I../../../

# Targets
TARGET = test_broker_queue
SRC = test_broker_queue.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean
