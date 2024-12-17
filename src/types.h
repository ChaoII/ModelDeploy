//
// Created by AC on 2024-12-17.
//

#pragma once

enum StatusCode {
    Success = 0x00,
    ModelInitFailed = 0x01,
    ModelPredictFailed = 0x02,
};

typedef struct {
    int x;
    int y;
} WPoint;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} WRect;


typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} WImage;

typedef struct {
    WPoint *points;
    size_t size;
} WPolygon;


typedef struct {
    WPolygon *boxes;
    char **texts;
    float *scores;
    size_t size;
} WOCRResult;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} WColor;

typedef struct {
    int width;
    int height;
} WSize;

typedef struct {
    WRect *boxes;
    int *label_ids;
    float *scores;
    size_t size;
} WDetectionResult;

