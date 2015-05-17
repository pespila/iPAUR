VERSION = 1.1
CC = g++
CPPFLAGS = -Wall -O2

DEPS = 	image.h \
		rgb.h \
		hsi.h \
		ycrcb.h \
		grayscale.h \
		type_conversion.h \
		robertscross.h \
		prewitt.h \
		sobel.h \
		create_filter.h \
		linear_filter.h \
		duto.h \
		dilatation.h \
		erosion.h \
		inverse.h \
		openclose.h \
		top_hats.h \
		laplacian_of_gaussian.h \
		gradient_filter.h \
		laplace.h \
		bluring.h \
		canny.h \
		median.h \
		arithmetic_functions.h \
		rgba.h \
		rof_model.h \
		analytical_operators.h

OBJ = 	main.o \
		image.o \
		rgb.o \
		hsi.o \
		ycrcb.o \
		grayscale.o \
		type_conversion.o \
		robertscross.o \
		prewitt.o \
		sobel.o \
		create_filter.o \
		linear_filter.o \
		duto.o \
		dilatation.o \
		erosion.o \
		inverse.o \
		openclose.o \
		top_hats.o \
		laplacian_of_gaussian.o \
		gradient_filter.o \
		laplace.o \
		bluring.o \
		canny.o \
		median.o \
		arithmetic_functions.o \
		rgba.o \
		rof_model.o \
		analytical_operators.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/usr/local/lib
IDFLAGS = -I/usr/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm *.o