VERSION = 1.1
CC = clang++
CPPFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall

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
		laplace.h \
		bluring.h \
		canny.h \
		median.h \
		arithmetic_functions.h \
		rgba.h \
		util.h \
		huber_rof_model.h \
		tv_l1_model.h \
		inpainting.h \
		util_color.h \
		huber_rof_model_color.h \
		tv_l1_model_color.h \
		inpainting_color.h

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
		laplace.o \
		bluring.o \
		canny.o \
		median.o \
		arithmetic_functions.o \
		rgba.o \
		util.o \
		huber_rof_model.o \
		tv_l1_model.o \
		inpainting.o \
		util_color.o \
		huber_rof_model_color.o \
		tv_l1_model_color.o \
		inpainting_color.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/opt/local/lib
IDFLAGS = -I/opt/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o