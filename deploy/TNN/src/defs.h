//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_DEFS_H
#define YOLOV6_DEFS_H

#ifndef YOLOV6_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define YOLOV6_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define YOLOV6_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
# define YOLOV6_WIN32
#elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
# define YOLOV6_UNIX
#endif


#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
# define NOMINMAX
#endif

#ifndef __unused
# define __unused
#endif

// debug mode
#define YOLOV6_DEBUG

#endif //YOLOV6_DEFS_H
