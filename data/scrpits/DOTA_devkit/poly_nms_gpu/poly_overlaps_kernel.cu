
#include "poly_overlaps.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include<algorithm>

using namespace std;

//##define CUDA_CHECK(condition)\
//
//  do {
//    cudaError_t error = condition;
//    if (error != cudaSuccess) {
//
//    }
//  }

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


#define maxn 510
const double eps=1E-8;

__device__ inline int sig(float d){
    return(d>eps)-(d<-eps);
}
// struct Point{
//     double x,y; Point(){}
//     Point(double x,double y):x(x),y(y){}
//     bool operator==(const Point&p)const{
//         return sig(x-p.x)==0&&sig(y-p.y)==0;
//     }
// };

__device__ inline int point_eq(const float2 a, const float2 b) {
    return (sig(a.x - b.x) == 0) && (sig(a.y - b.y)==0);
}

__device__ inline void point_swap(float2 *a, float2 *b) {
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2* last)
{
    while ((first!=last)&&(first!=--last)) {
        point_swap (first,last);
        ++first;
    }
}
// void point_reverse(Point* first, Point* last)
// {
//     while ((first!=last)&&(first!=--last)) {
//         point_swap (first,last);
//         ++first;
//     }
// }


__device__ inline float cross(float2 o,float2 a,float2 b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
__device__ inline float area(float2* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(float2 a,float2 b,float2 c,float2 d,float2&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}



//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
// __device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b){
//     // TODO: The static variable may be the reason, why single thread is ok, multiple threads are not work
//     printf("polygon_cut, offset\n");
    
//     static float2 pp[maxn];
//     int m=0;p[n]=p[0];
//     for(int i=0;i<n;i++){
//         if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
//         if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
//             lineCross(a,b,p[i],p[i+1],pp[m++]);
//     }
//     n=0;

//     for(int i=0;i<m;i++)
//         if(!i||!(point_eq(pp[i], pp[i-1])))
//             p[n++]=pp[i];
//     // while(n>1&&p[n-1]==p[0])n--;
//     while(n>1&&point_eq(p[n-1], p[0]))n--;
//     // int x = blockIdx.x * blockDim.x + threadIdx.x;
//     // // corresponding to k
//     // int y = blockIdx.y * blockDim.y + threadIdx.y;
//     // int offset = x * 1 + y;
//     // printf("polygon_cut, offset\n");
// }

__device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b, float2* pp){
    // TODO: The static variable may be the reason, why single thread is ok, multiple threads are not work
    // printf("polygon_cut, offset\n");
    
    // static float2 pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;

    for(int i=0;i<m;i++)
        if(!i||!(point_eq(pp[i], pp[i-1])))
            p[n++]=pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while(n>1&&point_eq(p[n-1], p[0]))n--;
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // // corresponding to k
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int offset = x * 1 + y;
    // printf("polygon_cut, offset\n");
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a,float2 b,float2 c,float2 d){
    float2 o = make_float2(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    // if(s1==-1) swap(a,b);
    // if(s2==-1) swap(c,d);
    // printf("before swap\n");
    // printf("a.x %f, a.y %f\n", a.x, a.y);
    // printf("b.x %f, b.y %f\n", b.x, b.y);
    if(s1 == -1) point_swap(&a, &b);
    // printf("a.x %f, a.y %f\n", a.x, a.y);
    // printf("b.x %f, b.y %f\n", b.x, b.y);
    // printf("after swap\n");
    if(s2 == -1) point_swap(&c, &d);
    float2 p[10]={o,a,b};
    int n=3;

    // // manually implement polygon_cut(p, n, a, b)
    // float2 pp[maxn];
    // // polygon_cut(p, n, o, c)
    // int m=0;p[n]=p[0];
    // for(int i=0;i<n;i++){
    //     if(sig(cross(o,c,p[i]))>0) pp[m++]=p[i];
    //     if(sig(cross(o,c,p[i]))!=sig(cross(o,c,p[i+1])))
    //         lineCross(o,c,p[i],p[i+1],pp[m++]);
    // }
    // n=0;

    // for(int i=0;i<m;i++)
    //     if(!i||!(point_eq(pp[i], pp[i-1])))
    //         p[n++]=pp[i];
    // while(n>1&&point_eq(p[n-1], p[0]))n--;

    // // polygon_cut(p, n, c, d)
    // m=0;p[n]=p[0];
    // for(int i=0;i<n;i++){
    //     if(sig(cross(c,d,p[i]))>0) pp[m++]=p[i];
    //     if(sig(cross(c,d,p[i]))!=sig(cross(c,d,p[i+1])))
    //         lineCross(c,d,p[i],p[i+1],pp[m++]);
    // }
    // n=0;

    // for(int i=0;i<m;i++)
    //     if(!i||!(point_eq(pp[i], pp[i-1])))
    //         p[n++]=pp[i];
    // while(n>1&&point_eq(p[n-1], p[0]))n--;

    // // polygon_cut(p, n, d, o)
    // m=0;p[n]=p[0];
    // for(int i=0;i<n;i++){
    //     if(sig(cross(d,o,p[i]))>0) pp[m++]=p[i];
    //     if(sig(cross(d,o,p[i]))!=sig(cross(d,o,p[i+1])))
    //         lineCross(d,o,p[i],p[i+1],pp[m++]);
    // }
    // n=0;

    // for(int i=0;i<m;i++)
    //     if(!i||!(point_eq(pp[i], pp[i-1])))
    //         p[n++]=pp[i];
    // while(n>1&&point_eq(p[n-1], p[0]))n--;
    float2 pp[maxn];
    polygon_cut(p,n,o,c,pp);
    polygon_cut(p,n,c,d,pp);
    polygon_cut(p,n,d,o,pp);
    float res=fabs(area(p,n));
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // corresponding to k
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x * 1 + y;
    // printf("intersectArea2, offset: %d, %f, %f, %f, %f, %f, %f, %f, %f, res: %f\n", offset, a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y, res);
    if(s1*s2==-1) res=-res;return res;

}
//求两多边形的交面积
// TODO: here changed the input, this need to be debug
__device__ inline float intersectArea(float2*ps1,int n1,float2*ps2,int n2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // corresponding to k
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x * 1 + y;
    if(area(ps1,n1)<0) point_reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) point_reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            // printf("offset: %d, %f, %f, %f, %f, %f, %f, %f, %f addArea: %f \n", 
            // offset, ps1[i].x, ps1[i].y, ps1[i + 1].x, ps1[i + 1].y, ps2[j].x, ps2[j].y, 
            // ps2[j + 1].x, ps2[j + 1].y, intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]));

            // float2 a = ps1[i];
            // float2 b = ps1[i+1];
            // float2 c = ps2[j];
            // float2 d = ps2[j+1];
            // res+=intersectArea2(a,b,c,d);
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}




//__device__ inline double iou_poly(vector<double> p, vector<double> q) {
//    Point ps1[maxn],ps2[maxn];
//    int n1 = 4;
//    int n2 = 4;
//    for (int i = 0; i < 4; i++) {
//        ps1[i].x = p[i * 2];
//        ps1[i].y = p[i * 2 + 1];
//
//        ps2[i].x = q[i * 2];
//        ps2[i].y = q[i * 2 + 1];
//    }
//    double inter_area = intersectArea(ps1, n1, ps2, n2);
//    double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
//    double iou = inter_area / union_area;
//
////    cout << "inter_area:" << inter_area << endl;
////    cout << "union_area:" << union_area << endl;
////    cout << "iou:" << iou << endl;
//
//    return iou;
//}

__device__ inline void RotBox2Poly(float const * const dbox, float2 * ps) {
    float cs = cos(dbox[4]);
    float ss = sin(dbox[4]);
    float w = dbox[2];
    float h = dbox[3];

    float x_ctr = dbox[0];
    float y_ctr = dbox[1];
    ps[0].x = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0);
    ps[1].x = x_ctr + cs * (w / 2.0) - ss * (h / 2.0);
    ps[2].x = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0);
    ps[3].x = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0);

    ps[0].y = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0);
    ps[1].y = y_ctr + ss * (w / 2.0) + cs * (h / 2.0);
    ps[2].y = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0);
    ps[3].y = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0);
}


__device__ inline float devPolyIoU(float const * const dbbox1, float const * const dbbox2) {


    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;




    RotBox2Poly(dbbox1, ps1);
    RotBox2Poly(dbbox2, ps2);

    // printf("ps1: %f, %f, %f, %f, %f, %f, %f, %f\n", ps1[0].x, ps1[0].y, ps1[1].x, ps1[1].y, ps1[2].x, ps1[2].y, ps1[3].x, ps1[3].y);
    // printf("ps2: %f, %f, %f, %f, %f, %f, %f, %f\n", ps2[0].x, ps2[0].y, ps2[1].x, ps2[1].y, ps2[2].x, ps2[2].y, ps2[3].x, ps2[3].y);
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    //printf("inter_area: %f \n", inter_area);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    //printf("before union_area\n");
    //printf("union_area: %f \n", union_area);
    float iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }
    // printf("iou: %f \n", iou);
    return iou;
}

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {

//   const int col_start = blockIdx.y;
//   const int row_start = blockIdx.x;
  
  // corresponding to n
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // corresponding to k
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < N) && (y < K)) {
    int offset = x * K + y;
    
    //printf
    // printf("offset: %d dbbox: %f %f %f %f %f\n", offset, (dev_boxes + x*5)[0],
    // (dev_boxes + x*5)[1], (dev_boxes + x*5)[2], (dev_boxes + x*5)[3], 
    // (dev_boxes + x*5)[4] );
    // printf("offset: %d dbbox: %f %f %f %f %f\n", offset, (dev_query_boxes + y*5)[0],
    // (dev_query_boxes + y*5)[1], (dev_query_boxes + y*5)[2], (dev_query_boxes + y*5)[3], 
    // (dev_query_boxes + y*5)[4] );
    
    dev_overlaps[offset] = devPolyIoU(dev_boxes + x * 5, dev_query_boxes + y * 5);
  } 
}


void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}


void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id) {

  _set_device(device_id);

  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;


  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        n * 5 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        n * 5 * sizeof(float),
                        cudaMemcpyHostToDevice));



  CUDA_CHECK(cudaMalloc(&query_boxes_dev,
                        k * 5 * sizeof(float)));



  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        k * 5 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&overlaps_dev,
                        n * k * sizeof(float)));


  if (true){}


  dim3 blocks(DIVUP(n, 32),
              DIVUP(k, 32));

  dim3 threads(32, 32);


  overlaps_kernel<<<blocks, threads>>>(n, k,
                                    boxes_dev,
                                    query_boxes_dev,
                                    overlaps_dev);

  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));


  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));

}
