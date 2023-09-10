#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <inttypes.h>


#define HiN (-1) // high number as a placeholder for unneeded labels
#define LaL (-1)   // last label in the stack; value is irrelevant
#define MC_IP 239 // multi cast IP address, where the switches send packets to
#define MAX_HOPS 6 // multi cast IP address, where the switches send packets to

#define LOGLEVEL 0
#define LOGLEVEL_EBPF 0
#define DO_DETAILED_MEASUREMENTS 0
// #define NUM_THREADS 8

#define FLAG_READ_HNSAS            1
#define FLAG_SET_HNSAS             2
#define FLAG_READ_DST_IPS          3
#define FLAG_HANDLE_NN_STAGE_1     4
#define FLAG_INPUT_PREP_STAGE_1    5
#define FLAG_FORWARD_STAGE_1       6
#define FLAG_ROUTE_PREP_STAGE_1    7
#define FLAG_INPUT_PREP_STAGE_2    8
#define FLAG_FORWARD_STAGE_2       9
#define FLAG_ROUTE_PREP_STAGE_2   10
#define FLAG_SINGLE_LOOP          11

#include <linux/bpf.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fstream>
#include <pthread.h>


struct bpf_elf_map {
  uint32_t type;
  uint32_t size_key;
  uint32_t size_value;
  uint32_t max_elem;
  uint32_t flags;
  uint32_t id;
  uint32_t pinning;
};


union bpf_attr attr = {
        .pathname = (uint64_t)(char*)"/sys/fs/bpf/tc/globals/",
        .bpf_fd = 0,
        .file_flags = 0
};


union bpf_attr map_attr = {
        .map_fd = 0,
        .key = 0,
        .value = 0
};


/*
 * Enough storage to keep four integer values for each of the
 * seven flags for five minutes assuming a sampling frequency of 1000Hz.
 * Format:
 *  <flag>     Identifier of the thing that was measured.
 *  <iter>     The iteration, monotonically increasing.
 *  <duration> Time it took in nanoseconds.
 *  <ns>       Number of elements that were processed during <duration>.
 */
struct measurement_tbl {
  int ncol = 4;
  int max_num_ts = 11 * 5 * 60 * 1000;
  int idx = 0;
  char filename[256] = "/home/sim/timings.bin";
  int * timings = nullptr;
};


struct thread_args {
  int thread_id = 0;
  struct NeuralNet * nn = nullptr;
  int * predictions = nullptr;
  float *buf1 = nullptr;
  float *buf2 = nullptr;
  int *intBuf = nullptr;
};


uint64_t get_nsecs() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000UL + ts.tv_nsec;
  // UL stands for unsigned long and makes sure, the 1e9 is read correctly
}


void printVector(float * data, int dim) {
  for(int i = 0; i < dim; i++) {
    std::cout << std::setprecision(2) << data[i];
    if(i < dim - 1) {
      std::cout << ", ";
    }
  }
}


void printVector(int * data, int dim) {
  for(int i = 0; i < dim; i++) {
    std::cout << data[i];
    if(i < dim - 1) {
      std::cout << ", ";
    }
  }
}


void printRowMajorMatrix(float * data, int nrows, int ncols) {
  std::cout << "Row major Matrix of shape (" << nrows << ", " << ncols << ")" << std::endl;
  for(int row = 0; row < nrows; row++) {
    for(int col = 0; col < ncols; col++) {
      std::cout << std::fixed << std::setw(7) << std::setprecision(2) << data[row * ncols + col];
      if(col < ncols - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


void printColumnMajorMatrix(float * data, int nrows, int ncols) {
  std::cout << "Column major Matrix of shape (" << nrows << ", " << ncols << ")" << std::endl;
  for(int row = 0; row < nrows; row++) {
    for(int col = 0; col < ncols; col++) {
      std::cout << std::fixed << std::setw(7) << std::setprecision(2) << data[col * nrows + row];
      if(col < ncols - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


int adjustDim(int dim_in, int stride) {
  if(dim_in % stride == 0) {
    return dim_in;
  } else {
    return dim_in + stride - (dim_in % stride);
  }
}


struct NnConfig {
  int final_fcn_0 = 102;
  int final_fcn_1 = 105;
  int dim_out = 9;
  int hlsa_dim = 128;
  // HLSA Attention Config.
  int hlsa_num_heads = 14;
  int hlsa_dim_fcn = 125;
  int hlsa_dim_hidden = 48;
  int hlsa_dim_out = 50;
  int hlsa_dim_k = 24;
  int hlsa_dim_v = 128;
  int hlsa_dim_q = 48;

  int neighbor_fcn = 90;
  int hlsa_out_neighbor_embd = 215;

  int dim_embedding = 24;

  // Pad the values to fit the avx lengths as necessary.
  int final_fcn_0_avx = 112;
  int final_fcn_1_avx = 112;
  int hlsa_dim_avx = 128;
  // HLSA Attention Config.
  int hlsa_dim_fcn_avx = 128;
  int hlsa_dim_hidden_avx = 48;
  int hlsa_dim_out_avx = 50 * 14 + 4; // Next divisible from 700 is 704
  int hlsa_dim_k_avx = 32;
  int hlsa_dim_v_avx = 128;
  int hlsa_dim_q_avx = 48;

  int neighbor_fcn_avx = 112;
  int hlsa_out_neighbor_embd_avx = 224;

  int dim_embedding_avx = 32;
};


struct NeuralNet {
  int numSamples = 0;
  int numSwitches = 80;
  int numParams = -1;
  uint32_t fd_dst_ips = 0;
  uint32_t fd_nn_lbls = 0;
  uint32_t fd_net_stt = 0;
  uint32_t fd_t_user = 0;
  uint64_t * intHnsas = nullptr;
  uint64_t * intDsts = nullptr;
  float * hnsas = nullptr;
  float * allIps = nullptr;
  float * dstIps = nullptr;
  float * switchIps = nullptr;
  float * queries = nullptr;
  float * nnParams = nullptr;
  struct NnConfig * config = nullptr;
};


void fillAllIps(float * allIps) {
  int a, b, c = 1, i = 0;
  int K = 8;
  // TOP-OF-THE-RACK-SWITCHES
  for(a = 0; a < K; a++){
    for(b = 0; b < K / 2; b++){
      int key = ( a << 16 ) | ( b << 8 ) | ( c ); // 10.a.b.c equals 10.pod.switch.1 which is 10.0.0.1 for tor0
      for(int j = 0; j < 24; j++) {
        allIps[i * 24 + j] = (float) ((key >> (23 - j)) & 1); // writing the most significant bit to the first entry of the tensor
      }
      i += 1;
    }
  }
  // AGGREGATION SWITCHES
  for(a = 0; a < K; a++){
    for(b = K / 2; b < K; b++){
      int key = ( a << 16 ) | ( b << 8 ) | ( c );
      for(int j = 0; j < 24; j++) {
        allIps[i * 24 + j] = (float) ((key >> (23 - j)) & 1);
      }
      i += 1;
    }
  }
  // CORE SWITCHES
  a = K;
  for(b = 1; b <= K/2; b++){
    for(c = 1; c <= K/2; c++){
      int key = ( a << 16 ) | ( b << 8 ) | ( c );
      for(int j = 0; j < 24; j++) {
        allIps[i * 24 + j] = (float) ((key >> (23 - j)) & 1);
      }
      i += 1;
    }
  }
  std::cout << "all_IPs tensor filled successfully" << std::endl;
}


void setHnsas(const uint64_t * msgContent, float *hnsas, const int numHnsas) {
  for(int i = 0; i < numHnsas; i++) {
    for(int j = 0; j < 64; j++) {
      hnsas[128 * i + 2 * j] = (float)((msgContent[i] >> (j)) & 1); // testen, ob das mit float funktioniert (es sollte am Ende float sein)
      hnsas[128 * i + 2 * j + 1] = 1 - (float)((msgContent[i] >> (j)) & 1);
    }
  }
}


void initHnsas(uint64_t * hnsas) {
  uint64_t initialHnsas[] = {
          5720976303843509535U,
          15131020779273914071U,
          15131091148051645687U,
          5905445312561742999U,
          15419211573040580343U,
          6195909904897476311U,
          15131016381262005463U,
          6193680095349900439U,
          15128799765819361015U,
          6193662503163856087U,
          15130972400763339991U,
          6193658105083791095U,
          15419281941752251607U,
          15419290729288893653U,
          15131033973414495479U,
          5905432127012144343U,
          15130981196889916599U,
          15131025177355027671U,
          15128768979461276887U,
          15419202768358671575U,
          15130985594935379095U,
          6195861526418359511U,
          5905427728964585207U,
          5905440923070564055U,
          15128799765819360983U,
          15131016381227402967U,
          15130985594936427735U,
          15130985594902873303U,
          5907617947504674007U,
          5905427728964585207U,
          15419220369134651095U,
          6193658105116296919U,
          5905427728931030775U,
          6195852721718101719U,
          6195848323671590421U,
          6195848323671590421U,
          6195914302976493271U,
          15128834950157895383U,
          15131038371494560983U,
          15131016381228451575U,
          5907609160035141335U,
          15131007585134380759U,
          15130981196889916631U,
          15419211564450645655U,
          15128843737694537431U,
          15419215971087090903U,
          15128839348237960407U,
          15130981196856362199U,
          6195843934233363671U,
          15131038371493512855U,
          15130998789075961047U,
          15131016381262005911U,
          15128839348237960343U,
          15128826154065921239U,
          15131086750005134999U,
          15128799765820409047U,
          6195931895163585751U,
          15131016381227402967U,
          5905432126977541847U,
          6193658105083791063U,
          15130998789074913015U,
          15128799765785806551U,
          5905449719197140695U,
          15128834950191449239U,
          5905432127011095767U,
          15419215971088139479U,
          5905427728931030775U,
          15131038371494560983U,
          15131073555866125463U,
          15131033973448049815U,
          6195909904929982167U,
          6195909904931030231U,
          15128834950157895383U,
          15130989992949384407U,
          15128843737695585495U,
          5905449719198188759U,
          15130998789074912503U,
          15131025168764045015U,
          15130998789074912983U,
          15130985594936427735U
  };
  for(int i = 0; i < 80; i++) {
    hnsas[i] = initialHnsas[i];
  }
}


int makeKey(int a, int b, int c, int i) {
  return ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | MC_IP;
}


int construct_file_descriptors(struct NeuralNet * nn) {
  char *path = (char*)"/sys/fs/bpf/tc/globals/dest_IPs";
  attr.pathname = (uint64_t)path;
  nn->fd_dst_ips = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(nn->fd_dst_ips == -1){
    std::cerr << "Map dest_IPs does not exist! Abort operation. \n";
    return EXIT_FAILURE;
  }

  path = (char*)"/sys/fs/bpf/tc/globals/nn_labels";
  attr.pathname = (uint64_t)path;
  nn->fd_nn_lbls = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(nn->fd_nn_lbls == -1){
    std::cerr << "Map nn_labels does not exist! Abort operation. \n";
    return EXIT_FAILURE;
  }

  path = (char*)"/sys/fs/bpf/tc/globals/network_state";
  attr.pathname = (uint64_t)path;
  nn->fd_net_stt = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(nn->fd_net_stt == -1){
    std::cerr << "Map network_state does not exist! Abort operation. \n";
    return EXIT_FAILURE;
  }

  path = (char*)"/sys/fs/bpf/tc/globals/t_user";
  attr.pathname = (uint64_t)path;
  nn->fd_t_user = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(nn->fd_t_user == -1){
    std::cerr << "Map t_user does not exist! Abort operation. \n";
    return EXIT_FAILURE;
  }

  std::cout << "file descriptors set successfully" << std::endl;
  return EXIT_SUCCESS;
}


int init_nn_labels_map(struct NeuralNet * nn, const int * labels, uint32_t dst_IP) {
  map_attr.map_fd = nn->fd_nn_lbls;

  uint32_t a = (dst_IP >> 8) 	& 0xff;
  uint32_t b = (dst_IP >> 16) 	& 0xff;
  uint32_t c = (dst_IP >> 24) 	& 0xff;
  uint64_t key, ret;
  __u32 n = 0;
  __u32 val[6];
  key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
  map_attr.key = (__u64)&key;
  map_attr.value = (__u64)&val[0];
  for(n=0; n<6; n++){
    val[n] = labels[n];
  }
  map_attr.value = (__u64)&val[0];
  ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){ // only then we want to fill the values
    ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr));
    if(ret == -1){
      std::cerr << "Updating process of map nn_labels failed! No labels for IP: 10." << a << "." << b << "." << c << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}


int init_nn_labels_eBPF(struct NeuralNet * nn) {
  int labels[MAX_HOPS];
  uint32_t own_pod = 0;
  uint32_t own_switch = 0;
  uint32_t own_host = 3;
  int K = 8;
  // initializes the values in the eBPF Map nn_labels with realistic fall-back routes
  uint32_t dst_pod, dst_switch, dst_host, j = 0;
  uint32_t key = ( dst_host << 24 ) | ( dst_switch << 16 ) | ( dst_pod << 8 ) | 10;
  for(dst_pod = 0; dst_pod < K; dst_pod++){
    for(dst_switch = 0; dst_switch < K/2; dst_switch++){
      for(dst_host = 2; dst_host < (K/2 + 2); dst_host++){
        key = ( dst_host << 24 ) | ( dst_switch << 16 ) | ( dst_pod << 8 ) | 10;
        if(own_pod == dst_pod){
          // same pod
          if(own_switch == dst_switch){
            // same ToR
            labels[0] = (int)dst_host - 1; // hosts 2-5 but labels 1-4
            labels[1] = LaL;
            for(j=2; j<6; j++){
              labels[j] = HiN;
            }
            if(EXIT_FAILURE == init_nn_labels_map(nn, labels,key)) { // set MPLS labels for this IP
              std::cerr << "Failed to initialize labels for hosts on same switch" << std::endl;
              return EXIT_FAILURE;
            }
          } else {
            // same pod other ToR
            labels[0] = (int)own_switch + 5; // the aggregation switch directly above the own host ip+4 and +1 to have a valid label
            labels[1] = (int)dst_switch + 1; // switches 0-3 but labels 1-4
            labels[2] = (int)dst_host - 1; // hosts 2-5 but labels 1-4
            labels[3] = LaL;
            for(j=4; j<6; j++){
              labels[j] = HiN;
            }
            if(EXIT_FAILURE == init_nn_labels_map(nn, labels, key)) { // set MPLS labels for this IP
              std::cerr << "Failed to initialize labels for hosts on other switches in same pod" << std::endl;
              return EXIT_FAILURE;
            }
          }
        } else {
          // other pod
          labels[0] = (int)own_switch + 5; // the aggregation switch directly above the own host ip+4 and +1 to have a valid label
          labels[1] = (int)own_host + 3; // [2-5] + 3 is [5-8] which are the labels, that lead to core switches
          labels[2] = (int)dst_pod + 1; // pods 0-7 but labels 1-8
          labels[3] = (int)dst_switch + 1; // switches 0-3 but labels 1-4
          labels[4] = (int)dst_host - 1; // hosts 2-5 but labels 1-4
          labels[5] = LaL;
          if(EXIT_FAILURE == init_nn_labels_map(nn, labels, key)) { // set MPLS labels for this IP
            std::cerr << "Failed to initialize labels for hosts in different pods" << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
    }
  }
  std::cout << "eBPF map nn labels successfully initialized" << std::endl;
  return EXIT_SUCCESS;
}


int readHnsasFromMap(struct NeuralNet * nn) {
  int k = 8;
  int idx = 0;
  int c = 1;
  int key;
  uint64_t val;
  map_attr.map_fd = nn->fd_net_stt; // needs to be in this function, otherwise it does not work

  for(int pod = 0; pod < k; pod++) {
    for (int tor = 0; tor < k / 2; tor++) {
      key = makeKey(pod, tor, 1, idx);
      map_attr.key = (uint64_t) &key;
      map_attr.value = (uint64_t) &val;
      long ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
      if (ret == -1) {
        std::cerr << "Reading process of map network_state failed! No information for IP: "
                  << MC_IP << "." << pod << "." << tor << "." << c << std::endl;
        return EXIT_FAILURE;
      }
      nn->intHnsas[idx] = val;
      idx += 1;
    }
  }
  for(int pod = 0; pod < k; pod++) {
    for(int agg = k / 2; agg < k; agg++) {
      key = makeKey(pod, agg, 1, idx);
      map_attr.key = (uint64_t)&key;
      map_attr.value = (uint64_t)&val;
      long ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
      if(ret == -1){
        std::cerr << "Reading process of map network_state failed! No information for IP: "
                  << MC_IP << "." <<  pod  << "." << agg  << "." << c << std::endl;
        return EXIT_FAILURE;
      }
      nn->intHnsas[idx] = val;
      idx += 1;
    }
  }
  for(int g1 = 1; g1 <= k / 2; g1++) {
    for(int g2 = 1; g2 <= k / 2; g2++) {
      key = makeKey(k, g1, g2, idx);
      map_attr.key = (uint64_t)&key;
      map_attr.value = (uint64_t)&val;
      long ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
      if(ret == -1){
        std::cerr << "Reading process of map network_state failed! No information for IP: "
                  << MC_IP << "." <<  k  << "." << g1  << "." << g2 << std::endl;
        return EXIT_FAILURE;
      }
      nn->intHnsas[idx] = val;
      idx += 1;
    }
  }
  return EXIT_SUCCESS;
}


void gatherDstIps(struct NeuralNet *nn) {
  int K = 8;
  int counter = 0;
  uint32_t n = 0;
  uint64_t val, key;
  long ret;
  map_attr.map_fd = nn->fd_dst_ips; // set file descriptor
  uint64_t current_time = get_nsecs();

  // TODO: Iterate over the map instead of checking each and every key.
  for(uint64_t a = 0; a < K; a++){
    for(uint64_t b = 0; b < K/2; b++){
      for(uint64_t c = 2; c < (K/2 + 2); c++){
        key = (c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
        map_attr.key = (uint64_t)&key;
        map_attr.value = (uint64_t)&val;
        ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
        if(ret == -1){
          // printf("Reading process of map dest_IPs failed! No information for IP: %d.%d.%d.%d. \n", 10, a, b, c);
          continue;
        }
        if(LOGLEVEL_EBPF > 0) { std::cout << "\tFound destination 10." << a << "." << b << "." << c << " - key is " << key << " - age is " << (double)(current_time - val) / 1e9 << std::endl; }
        // if(val  != 0){
        //   printf("For IP: %d.%d.%d.%d, the value is: %lld. \n", 10, a, b, c, val);
        // }
        if((current_time - val) < 10000000000 || 1) { // time in ns // 10s for testing
          // if there was a connection to this host in the last 10s
          nn->intDsts[counter] = key;
          counter += 1;
        }
      }
    }
  }
  nn->numSamples = counter;
}


int update_nn_labels(struct NeuralNet * nn, const int * labels, uint64_t dst_IP) {
  map_attr.map_fd = nn->fd_nn_lbls;

  uint64_t a = (dst_IP >> 8) 	  & 0xff;
  uint64_t b = (dst_IP >> 16) 	& 0xff;
  uint64_t c = (dst_IP >> 24) 	& 0xff;
  if(LOGLEVEL_EBPF > 0) {
    std::cout << "\t\tUpdate label for key " << dst_IP << " - IP is " << 10 << "." <<  a  << "." << b  << "." << c << std::endl;
  }
  uint64_t key, ret;
  uint32_t n = 0;
  uint32_t val[6];
  key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
  map_attr.key = (uint64_t)&key;
  for(n=0; n<6; n++){
    val[n] = 0;
  }
  map_attr.value = (uint64_t)&val[0];
  ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    std::cerr << "Reading process of map nn_labels failed! No information for IP: "
              << 10 << "." <<  a  << "." << b  << "." << c << std::endl;
    return EXIT_FAILURE;
  }
  if(LOGLEVEL_EBPF > 0) { //std::cout << "Gather the dsetination IPS" << std::endl; }
    printf("[mpls label] IP: %d.%d.%d.%d labels: %d %d %d %d %d %d \n", 10, a, b, c, val[0], val[1], val[2], val[3],
           val[4], val[5]);
  }
  for(n = 0; n < 6; n++){
    val[n] = labels[n];
  }
  ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    std::cerr << "Reading process of map nn_labels failed! No labels for IP: "
              << 10 << "." <<  a  << "." << b  << "." << c << std::endl;
    return EXIT_FAILURE;
  }
  // printf("[mpls label update] IP: %d.%d.%d.%d labels: %d %d %d %d %d %d \n", 10, a, b, c, val[0], val[1], val[2], val[3], val[4], val[5]);
  // std::cout << "eBPF map nn labels updated successfully" << std::endl;
  //std::cout << "\t\t--> updated" << std::endl;
  return EXIT_SUCCESS;
}


int updateLabelsNnStageOne(struct NeuralNet * nn, const int * predictions, int * chosenAggs) {
  uint64_t dst_pod, dst_switch, dst_host;
  uint64_t ownPod = 0;
  int labels[MAX_HOPS];
  int count = 0;
  for(int i = 0; i < nn->numSamples; i++) {
    dst_pod     = (nn->intDsts[i] >> 8) 	& 0xff; // pod
    dst_switch  = (nn->intDsts[i] >> 16) 	& 0xff; // switch
    dst_host    = (nn->intDsts[i] >> 24) 	& 0xff; // host
    if(dst_pod == ownPod) {
      labels[0] = predictions[i];
      labels[1] = (int)(dst_switch + 1);
      labels[2] = (int)(dst_host - 1);
      labels[3] = LaL;
      labels[4] = HiN;
      labels[5] = HiN;
      if(EXIT_FAILURE == update_nn_labels(nn, labels, nn->intDsts[i])) {
        std::cerr << "Failed to update labels at stage one for IP: 10." << dst_pod << "." << dst_switch << "." << dst_host << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      nn->intDsts[count] = nn->intDsts[i];
      // Store the index of this node together with the chosen aggregation switch. Iterating
      // over this allows the retrieval of the indices, which in turn allows direct memcopy of
      // the inputs.
      chosenAggs[2 * count] = i;
      chosenAggs[2 * count + 1] = predictions[i];
      count += 1;
    }
  }
  nn->numSamples = count;
  return EXIT_SUCCESS;
}


int updateLabelsNnStageTwo(struct NeuralNet * nn, const int * predictions, const int * chosenAggs) {
  uint64_t dst_pod, dst_switch, dst_host;
  uint64_t ownPod = 0;
  int labels[MAX_HOPS];
  int count = 0;
  for(int i = 0; i < nn->numSamples; i++) {
    dst_pod     = (nn->intDsts[i] >> 8) 	& 0xff; // pod
    dst_switch  = (nn->intDsts[i] >> 16) 	& 0xff; // switch
    dst_host    = (nn->intDsts[i] >> 24) 	& 0xff; // host
    labels[0] = chosenAggs[2 * i + 1];
    labels[1] = predictions[i];
    labels[2] = (int)dst_pod + 1; // pods 0-7 but labels 1-8
    labels[3] = (int)dst_switch + 1;
    labels[4] = (int)dst_host - 1;
    labels[5] = LaL;
    // std::cout << "Send packet over route " << labels[0] << " "<< labels[1] << " "<< labels[2] << " "<< labels[3] << " "<< labels[4] << " "<< labels[5] << std::endl;
    if(EXIT_FAILURE == update_nn_labels(nn, labels, nn->intDsts[i])) {
      std::cerr << "Failed to update labels at stage two for IP: 10." << dst_pod << "." << dst_switch << "." << dst_host << std::endl;
      return EXIT_FAILURE;
    }
  }
  nn->numSamples = 0;
  return EXIT_SUCCESS;
}


int handle_NN_pt1(struct NeuralNet * nn) {
  /*
   * Iterate over the destination IP addresses in integer form. Check if
   * any of the addresses belong to hosts under the same ToR. If so, update
   * their labels accordingly and update them.
   * Keep all other destination IPs and adjust the number of samples accordingly.
   *
   * The function stores the remaining destinations back in intDsts.
   */
  int labels[MAX_HOPS];
  if(nn->numSamples == 0){
    return EXIT_SUCCESS;
  }
  int i, j, n = 0;
  uint64_t own_pod = 0;
  uint64_t own_switch = 0;

  for(i = 0; i < nn->numSamples; i++){
    if(LOGLEVEL_EBPF > 0) { std::cout << "Update label for host " << i << " of " << nn->numSamples << std::endl; }
    uint64_t dst_pod     = (nn->intDsts[i] >> 8) 	  & 0xff; // pod
    uint64_t dst_switch  = (nn->intDsts[i] >> 16) 	& 0xff; // switch
    uint64_t dst_host    = (nn->intDsts[i] >> 24) 	& 0xff; // host
    if(dst_switch == own_switch && dst_pod == own_pod){
      // only one ToR switch is needed
      labels[0] = (int)dst_host - 1; // hosts 2-5 but labels 1-4
      labels[1] = LaL;
      for(j = 2; j < 6; j++){
        labels[j] = HiN;
      }
      if(update_nn_labels(nn, labels, nn->intDsts[i]) == EXIT_FAILURE) {// set MPLS labels for this IP
        std::cerr << "Failed to update labels in handle_NN_pt1" << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      nn->intDsts[n] = nn->intDsts[i];
      n += 1;
    }
  }
  nn->numSamples = n;
  return EXIT_SUCCESS;
}


void binarizeIntIp(float * buf, uint64_t ip) {
  for(int j = 0; j < 8; j++) {
    buf[j] = (float)((ip >> (15 - j)) & 1);
  }
  for(int j = 0; j < 8; j++) {
    buf[j + 8] = (float) ((ip >> (23 - j)) & 1); //similar to the second octet, now the bits from 23 back to 16
  }
  for(int j = 0; j < 8; j++) {
    buf[j + 16] = (float) ((ip >> (31 - j)) & 1); // as before, the bits form 31 to 24
  }
}


void prepareInputIpsStageOne(struct NeuralNet * nn) {
  // Prepare the destination and current location inputs.
  for(int i = 0; i < nn->numSamples; i++) {
    // From the integer representation of the destination IP create a binary vector.
    // Add one row to the input marix in each iteration through ofsetting the start pointer.
    binarizeIntIp(nn->dstIps + i * nn->config->dim_embedding, nn->intDsts[i]);
    // Copy the switch IP of the first ToR switch to the switchIps pointer. Copy once for
    // each sample, i.e., create one row in the input tensor.
    memcpy(nn->switchIps + i * nn->config->dim_embedding,
           nn->allIps,
           nn->config->dim_embedding * sizeof(float)
    );
    // Copy the switch IP to the attention queries.
    memcpy(nn->queries + i * nn->config->hlsa_dim_q,
           nn->switchIps + i * nn->config->dim_embedding,
           nn->config->dim_embedding * sizeof(float)
    );
    // Copy the destination IP to the attention queries, offset additionally by the embedding length.
    memcpy(nn->queries + i * nn->config->hlsa_dim_q + nn->config->dim_embedding,
           nn->dstIps + i * nn->config->dim_embedding,
           nn->config->dim_embedding * sizeof(float)
    );
  }
}


void prepareInputIpsStageTwo(struct NeuralNet * nn, const int * chosenAggs) {
  // Prepare the destination and current location inputs.
  for(int i = 0; i < nn->numSamples; i++) {
    // Copy the old destination IP to the beginning of the input.
    memcpy(
            nn->dstIps + i * nn->config->dim_embedding,
            nn->dstIps + chosenAggs[2 * i] * nn->config->dim_embedding,
            nn->config->dim_embedding * sizeof(float)
    );
    // Copy the switch IP of the first ToR switch to the switchIps pointer. Copy once for
    // each sample, i.e., create one row in the input tensor.
    memcpy(nn->switchIps + i * nn->config->dim_embedding,
           nn->switchIps + chosenAggs[2 * i] * nn->config->dim_embedding,
           nn->config->dim_embedding * sizeof(float)
    );
    // Copy the queries to the beginning of the tensor.
    memcpy(nn->queries + i * nn->config->hlsa_dim_q,
           nn->queries + chosenAggs[2 * i] * nn->config->hlsa_dim_q,
           nn->config->hlsa_dim_q * sizeof(float)
    );
  }
}


int readParams(char * filename, float* params, int numParams) {
  FILE *file;
  file = fopen(filename, "rb");
  if(file == nullptr) {
    std::cerr << "could not open the parameter file " << std::string(filename) << std::endl;
    return -1;
  }
  if(fread(params, sizeof(float), numParams, file) != numParams) {
    std::cerr << "Could not read " << numParams << " into the provided buffer from file " << std::string(filename) << std::endl;
    return -2;
  }
  fclose(file);
  return 0;
}


int calcNumParams(NnConfig * config) {
  int num_params = 0;
  num_params += 2 * (config->final_fcn_1 * config->dim_out + config->dim_out);
  num_params += config->final_fcn_0 * config->final_fcn_1 + config->final_fcn_1;
  num_params += (config->hlsa_dim_fcn + config->neighbor_fcn) * config->final_fcn_0 + config->final_fcn_0;

  num_params += config->hlsa_num_heads * config->hlsa_dim_k * config->hlsa_dim_hidden;
  num_params += config->hlsa_num_heads * config->hlsa_dim_q * config->hlsa_dim_hidden;
  num_params += config->hlsa_num_heads * config->hlsa_dim_v * config->hlsa_dim_out;
  num_params += config->hlsa_num_heads * config->hlsa_dim_out * config->hlsa_dim_fcn + config->hlsa_dim_fcn;

  num_params += 2 * config->dim_embedding * config->neighbor_fcn + config->neighbor_fcn;
  return num_params;
}


void testCalcNumParams() {
  int truth = 264774;
  struct NnConfig config;
  int returned = calcNumParams(&config);
  if(truth != returned) {
    std::cerr << "Test testCalcNumParams failed, got " << returned << " instead of " << truth << std::endl;
  }
}


void adaptNnConfigToAvx(NnConfig *config, int factor) {
  // Pad the original configuration to multiples of factor, i.e., such that
  // the numbes are evenly divisible by factor.
  config->final_fcn_0     += config->final_fcn_0     % factor;
  config->final_fcn_1     += config->final_fcn_1     % factor;
  config->dim_out         += config->dim_out         % factor;
  config->hlsa_dim_fcn    += config->hlsa_dim_fcn    % factor;
  config->hlsa_dim_hidden += config->hlsa_dim_hidden % factor;
  config->hlsa_dim_out    += config->hlsa_dim_out    % factor;
  config->neighbor_fcn    += config->neighbor_fcn    % factor;
  config->dim_embedding   += config->dim_embedding   % factor;
  config->hlsa_dim        += config->hlsa_dim        % factor;
}


float * reluGemmAvx(float * w, float * x, float * y, int bs, int dim_in, int dim_out) {
  __m512 zero = _mm512_set1_ps(0);
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(w[dim_in * dim_out + k] / 16);
      for(int j = 0; j < dim_in; j+=16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in + j));
        __m512 x_part = _mm512_loadu_ps(x + (i * dim_in + j));
        x_part = _mm512_max_ps(x_part, zero);

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * dim_out + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in * dim_out + dim_out;
}


float * gemmavx(float * w, float * x, float * y, int bs, int dim_in, int dim_in_avx, int dim_out) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(w[dim_in_avx * dim_out + k] / 16);
      for(int j = 0; j < dim_in; j += 16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in_avx + j));
        __m512 x_part = _mm512_loadu_ps(x + (i * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * dim_out + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in_avx * dim_out + dim_out;
}


float * spacedOutputGemmavx(float * w, float * x, float * y, int bs, int dim_in, int dim_in_avx, int dim_out, int out_spacing) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of out_spacing. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  // The additional dim_in_avx is needed to offset the weights accordingly. The weights
  // contain padding that cancel out the space that we read beyond the actual input
  // because of reading always 16 floats. At the same time, we need the original
  // input dimension to advance the x vector accordingly.
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(w[dim_in_avx * dim_out + k] / 16);
      for(int j = 0; j < dim_in; j += 16) {
        // if((i == 0 || i == 1) && k == 0) {
        //   std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        //   std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        // }
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in_avx + j));
        __m512 x_part = _mm512_loadu_ps(x + (i * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * out_spacing + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in_avx * dim_out + dim_out;
}


float * spacedOutputGemmavxNoBias(float * w, float * x, float * y, int bs, int dim_in, int dim_in_avx, int dim_out, int out_spacing) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of out_spacing. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(0);
      for(int j = 0; j < dim_in; j += 16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in_avx + j));
        __m512 x_part = _mm512_loadu_ps(x + (i * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * out_spacing + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in_avx * dim_out;
}


float * slicedGemmavx(float * w, float * x, float * y, const int * indices, int bs, int dim_in, int dim_out, int spacing) {
  // Implements a linear layer that computes x[idx, :]A + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of dim_out. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  int row_idx;
  for(int i = 0; i < bs; i++) {
    row_idx = indices[i];
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(w[dim_in * dim_out + k] / 16);
      for(int j = 0; j < dim_in; j+=16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in + j));
        __m512 x_part = _mm512_loadu_ps(x + (row_idx * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * spacing + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in * dim_out + dim_out;
}


float * slicedGemmavxNoBias(float * w, float * x, float * y, const int * indices, int bs, int dim_in, int dim_in_avx, int dim_out, int spacing) {
  // Implements a linear layer that computes x[idx, :]A + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of dim_out. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  int row_idx;
  for(int i = 0; i < bs; i++) {
    row_idx = indices[i];
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(0);
      for(int j = 0; j < dim_in; j += 16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in_avx + j));
        __m512 x_part = _mm512_loadu_ps(x + (row_idx * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * spacing + k] = _mm512_reduce_add_ps(sumx16);
    }
  }
  return w + dim_in_avx * dim_out;
}


void outerProduct(const float * a, const float * b, float * y, int nrows_a, int nrows_b, int ncols) {
  // Computes an outer product between a and b. The matrix a has a shape
  // of (nrows_a, ncols). The matrix b of (nrows_b, ncols). Both memory
  // layout of both matrices is column major.
  // The computed matrix has a shape of (nrows_a, nrows_b) with a column major
  // memory layout.
  for(int i = 0; i < nrows_a; i++) {
    for(int j = 0; j < nrows_b; j++) {
      y[j * nrows_a + i] = 0;
      for(int k = 0; k < ncols; k++) {
        // std::cout << i << " " << j << " " << k << " | ";
        // std::cout << "y[" << j * nrows_a + i << "] += a[" << i * ncols + k << "] * b[" << j * ncols + k << "]";
        // std::cout << std::endl;
        y[j * nrows_a + i] += a[i * ncols + k] * b[j * ncols + k];
      }
    }
  }
}


void outerProductAvx(float * a, float * b, float * y, int nrows_a, int nrows_b, int ncols) {
  // Computes an outer product between a and b. The matrix a has a shape
  // of (nrows_a, ncols). The matrix b of (nrows_b, ncols). Both memory
  // layout of both matrices is column major.
  // The computed matrix has a shape of (nrows_a, nrows_b) with a column major
  // memory layout.
  for(int i = 0; i < nrows_a; i++) {
    for(int j = 0; j < nrows_b; j++) {
      __m512 sumx16 = _mm512_set1_ps(0);
      for(int k = 0; k < ncols; k+=16) {
        // std::cout << i << " " << j << " " << k << " | ";
        // std::cout << "y[" << j * nrows_a + i << "] += a[" << i * ncols + k << "] * b[" << j * ncols + k << "]";
        // std::cout << std::endl;
        __m512 a_part = _mm512_loadu_ps(a + (i * ncols + k));
        __m512 b_part = _mm512_loadu_ps(b + (j * ncols + k));
        sumx16 = _mm512_fmadd_ps(a_part, b_part, sumx16);
      }
      y[j * nrows_a + i] = _mm512_reduce_add_ps(sumx16);
    }
  }
}


void testOuterProduct() {
  int passed = 1, passed_avx = 1;
  auto y = (float*)calloc(10 * 5, sizeof(float));
  float a[] = {
          -1.5255959033966064, -0.7502318024635315, -0.6539809107780457, -1.6094847917556763, -0.1001671776175499, -0.6091889142990112, -0.9797722697257996, -1.6090962886810303, -0.7121446132659912, 0.30372199416160583, -0.777314305305481, -0.25145524740219116, -0.22227048873901367, 1.6871134042739868, 0.22842517495155334, 0.46763551235198975, -0.6969724297523499, -1.1607614755630493, 0.6995424032211304, 0.1990816295146942, 0.8656923770904541, 0.2444038987159729, -0.6629113554954529, 0.8073082566261292, 1.1016806364059448, -0.1759360432624817, -2.2455577850341797, -1.4464579820632935, 0.0611552819609642, -0.6177444458007812, -0.7980698347091675, -0.13162320852279663,
          1.8793457746505737, -0.07213178277015686, 0.15777060389518738, -0.7734549045562744, 0.1990565061569214, 0.04570277780294418, 0.15295691788196564, -0.47567880153656006, -0.11101982742547989, 0.2927352488040924, -0.1578451544046402, -0.028787139803171158, 2.3571109771728516, -1.0373387336730957, 1.5747981071472168, -0.6298472285270691, -0.9273917078971863, 0.5451415181159973, 0.06628026068210602, -0.4370401203632355, 0.7626006007194519, 0.4415109157562256, 1.1651384830474854, 2.0153918266296387, 0.13741245865821838, 0.9386447072029114, -0.18600109219551086, -0.6446393132209778, 1.539245843887329, -0.8695876002311707, -3.331153631210327, -0.7478722333908081,
          -0.025502461940050125, -1.023330569267273, -0.5961851477622986, -1.0055307149887085, -0.21060630679130554, -0.007547527551651001, 1.6734272241592407, 0.010342830792069435, -0.703956663608551, -0.18526579439640045, -0.9962350726127625, -0.8312552571296692, -0.4610220193862915, -0.5600824356079102, 0.3955761790275574, -0.9822770953178406, -0.5064865946769714, 0.09977540373802185, -0.653973400592804, 0.731693685054779, -1.434385895729065, -0.5008130669593811, 0.17163313925266266, -0.15999312698841095, 0.25463348627090454, -0.5019572973251343, -1.041200041770935, 0.7322672009468079, -1.048340082168579, -0.47087720036506653, 0.29113635420799255, 1.9907042980194092,
          0.6614453196525574, 1.1899205446243286, 0.8165339231491089, -0.9135236144065857, 1.385145664215088, -0.813846230506897, -0.9275765419006348, 1.1119632720947266, 1.3352056741714478, 0.6042736172676086, -0.10344208031892776, -0.15121692419052124, -2.1020829677581787, -0.6200219392776489, -1.4782309532165527, -1.1334174871444702, 0.873796284198761, -0.5602594017982483, 1.2857844829559326, 0.8168238401412964, 0.2053041011095047, 0.3051071763038635, 0.5356870293617249, -0.4311850070953369, 2.558138370513916, -0.23336388170719147, -0.013472129590809345, 1.8606348037719727, -1.9804062843322754, 1.798582911491394, 0.10181158781051636, 0.3400059938430786,
          0.7123645544052124, -1.7765072584152222, 0.3538645803928375, 1.1996132135391235, -0.3029974102973938, -1.7618416547775269, 0.6348446011543274, -0.8043590784072876, -1.6111117601394653, -1.8716129064559937, 0.5430836081504822, 0.6606786251068115, 2.2952115535736084, 0.6749059557914734, 1.713321566581726, -1.7942733764648438, -1.363267183303833, -0.9832196235656738, 1.5112667083740234, 0.6418707370758057, 0.472963809967041, -0.4285900890827179, 0.5513707399368286, -1.5473709106445312, 0.5181121230125427, 0.10653535276651382, 0.26924076676368713, 1.3247679471969604, 1.7460191249847412, 1.8549690246582031, -0.7063691020011902, 2.557086229324341,
          0.4175342917442322, -0.21271860599517822, -0.8399580121040344, -0.42001786828041077, -0.6240363121032715, -0.9772961139678955, 0.8748428225517273, 0.9872813820838928, 0.3095763325691223, 1.5206899642944336, 1.2052339315414429, -1.8155909776687622, -0.4034615457057953, -0.959145188331604, -0.005207703914493322, -0.07886313647031784, 0.8436542749404907, 1.1657012701034546, 0.5269321799278259, 1.6192532777786255, -0.963976263999939, 0.14152038097381592, -0.1636609584093094, -0.3582225739955902, 1.7222793102264404, -0.3035756051540375, 0.23887419700622559, 1.3440011739730835, 0.1032256931066513, 1.1003541946411133, -0.3416801989078522, 0.947338879108429,
          -0.568515956401825, 0.8375961780548096, 1.783660650253296, -0.1954246610403061, 0.5149161219596863, -1.8474775552749634, -2.9167425632476807, -0.5673298835754395, -0.541280210018158, 0.8951740264892578, -0.8825070261955261, 0.5318112373352051, -1.5457772016525269, -0.17329981923103333, 0.7282463312149048, 0.05706102028489113, 0.9055172204971313, 1.0462948083877563, -0.520596981048584, 1.3547837734222412, 0.235193133354187, 1.9142433404922485, 1.8364111185073853, 1.324532389640808, -0.9690091609954834, 1.2516363859176636, 1.2103241682052612, -0.5279206037521362, 0.2185661494731903, -0.5743072628974915, 1.4571250677108765, 1.7709556818008423,
          1.6499137878417969, -0.43200457096099854, -0.2710269093513489, -1.4391626119613647, 1.2470403909683228, 1.2738511562347412, 0.3909492492675781, 0.387210488319397, -0.07982871681451797, 0.34172430634498596, 0.94882732629776, -1.3839359283447266, 1.7240862846374512, -2.364765167236328, -0.9294909238815308, 0.2936252951622009, 0.21513202786445618, 0.9384636878967285, 1.4657076597213745, -0.5564743876457214, -0.7448408007621765, -0.20215721428394318, -0.22966790199279785, 0.0013313365634530783, 0.3752759099006653, -0.5810679197311401, -0.5723088383674622, 1.0097174644470215, -0.10564938932657242, -1.179695963859558, -0.09077959507703781, 0.5631143450737,
          -1.256014108657837, 0.8955550193786621, 0.16747736930847168, 0.7514208555221558, 2.4142298698425293, 1.020583987236023, -0.44048380851745605, -1.7341676950454712, -1.2362250089645386, 1.5785813331604004, -1.1160507202148438, 0.7677702307701111, -0.5882067680358887, 2.1188902854919434, -0.5421902537345886, -2.459254741668701, -1.1108287572860718, -1.1187208890914917, 0.7579955458641052, -0.49565765261650085, -0.19700005650520325, -0.033396217972040176, 0.7192915081977844, 1.064414620399475, 0.8340254426002502, -1.9162163734436035, -0.34202927350997925, -0.6604920625686646, 0.31508535146713257, 1.1422518491744995, 0.3055056631565094, -0.5788817405700684,
          -0.23828251659870148, -1.354174256324768, 0.2686893939971924, 0.11455696821212769, -1.5562971830368042, -1.0757436752319336, -0.8751946091651917, -0.4728187620639801, 0.9912368059158325, -0.05862228199839592, 1.1787645816802979, 0.6221849918365479, 0.7878500819206238, 1.368552327156067, -0.8506898283958435, 0.5126074552536011, 1.0476324558258057, -0.3175846338272095, 0.13948506116867065, 2.3402624130249023, -0.611609160900116, 0.8160271048545837, 0.24772299826145172, -0.3867267072200775, 0.19948451220989227, 0.7992695569992065, -0.26190340518951416, 0.1513296216726303, 1.1981666088104248, -2.2832581996917725, -1.012959361076355, -0.8878908753395081
  };
  float b[] = {
          0.6522192358970642, -0.8726202845573425, 0.035253752022981644, -0.33653029799461365, 1.4023319482803345, 0.4841214120388031, -0.7030450701713562, -0.8267660737037659, 0.7743960022926331, 0.6919939517974854, -1.0184799432754517, -0.8033716678619385, -0.7071132063865662, 0.7521182894706726, -0.019208278506994247, 1.1033329963684082, -0.606792151927948, -0.5252234935760498, -0.5661877393722534, 0.0006603985675610602, 0.7224587798118591, 0.15263520181179047, 0.14495977759361267, -2.344219446182251, 0.3600029945373535, 0.46668174862861633, 1.2830665111541748, 1.2678006887435913, 0.19883295893669128, 0.5440877079963684, -0.397816926240921, -1.929105520248413,
          0.23236869275569916, 0.8614656329154968, 0.6217573285102844, -1.7811895608901978, -0.7820609211921692, -1.4236700534820557, 1.6090764999389648, -0.03278759494423866, 0.8532333970069885, 0.055063650012016296, -1.7425371408462524, 0.8750037550926208, -2.7188172340393066, -0.22192060947418213, 0.34208494424819946, 1.1093477010726929, -0.5731475949287415, 0.9577845931053162, 0.000982023193500936, -1.3847686052322388, -0.9965022802352905, 0.8073481321334839, 1.1738862991333008, -0.9398464560508728, 1.310918927192688, -0.31670692563056946, -0.18610410392284393, -0.5764601826667786, 0.6866518259048462, 0.42086705565452576, -1.0213807821273804, 0.9885666370391846,
          -0.5618716478347778, -0.15792575478553772, 1.5042593479156494, -1.3950295448303223, 0.8007909655570984, -0.6619443893432617, 1.2563107013702393, 0.4999944567680359, -0.2713380753993988, 1.8469072580337524, -0.031249959021806717, -0.09387270361185074, -0.619074285030365, -0.6363265514373779, -0.42415860295295715, -2.0271668434143066, 0.40962907671928406, -1.5421266555786133, -1.0128618478775024, -0.02973751351237297, -0.2889522612094879, 0.15219318866729736, -0.29803404211997986, -0.13135384023189545, -0.628098726272583, 1.1968798637390137, 0.610993504524231, -0.4547743797302246, -0.9603701829910278, 0.2769045829772949, -0.680108904838562, -0.545787513256073,
          -0.4551834166049957, 0.3185957968235016, -0.35494208335876465, 0.6858943700790405, -0.37613728642463684, -2.4106996059417725, -1.2778087854385376, -0.06288741528987885, -0.0947127640247345, -2.3144304752349854, 0.5565339922904968, 0.5056920647621155, -0.2075958400964737, 0.6936318278312683, 0.41949039697647095, 2.252354383468628, 0.9385231137275696, 1.425292730331421, 1.5083258152008057, 0.10539496690034866, -1.6049960851669312, -0.10644838958978653, 0.2465665489435196, 0.6125083565711975, 0.739801824092865, -0.17860014736652374, 0.0784900113940239, -0.43981805443763733, -0.3607933819293976, -1.2617405652999878, 1.9146918058395386, -1.8612741231918335,
          -0.009674912318587303, 0.2603876292705536, 0.2820335328578949, 0.25829946994781494, -0.42654868960380554, 0.9807512164115906, 1.8588889837265015, -1.0920146703720093, 0.7630020380020142, 0.22761525213718414, -1.4569789171218872, 1.7043737173080444, -3.2686386108398438, 0.47498711943626404, -2.1142473220825195, -1.500230073928833, 1.0692973136901855, 1.4393831491470337, 0.5064594149589539, 0.8359752893447876, 1.1752967834472656, -0.34211742877960205, -0.3871636688709259, 0.5476537942886353, -0.15891987085342407, -0.7360489368438721, -0.2335187792778015, -0.5403915047645569, 0.15708433091640472, -0.5976229906082153, -0.8839093446731567, 0.6076730489730835
  };
  float c[] = {
          -0.49025821685791016, -2.3963510990142822, -5.525833606719971, 5.849405288696289, -2.752423048019409, -1.0735973119735718, -3.2683801651000977, -1.9998419284820557, 1.839798927307129, 0.872183084487915,
          2.280749559402466, 0.09631854295730591, 5.987659454345703, 8.065335273742676, -3.831732988357544, 4.238704204559326, 5.1552605628967285, -4.698294639587402, -2.6605422496795654, -4.972365856170654,
          -1.2172138690948486, 0.8748531341552734, 3.4983882904052734, 7.557455539703369, -5.449447154998779, 1.1133397817611694, 2.7568955421447754, -2.214409589767456, 4.660851955413818, -2.808370351791382,
          2.0770440101623535, -8.581881523132324, -6.8964996337890625, -2.4327359199523926, -5.941349506378174, -1.7906639575958252, 8.080954551696777, -3.475444793701172, -8.394039154052734, 10.368856430053711,
          1.1754107475280762, -6.05797004699707, 3.2808265686035156, 7.07057523727417, -10.557694435119629, -1.534283995628357, 1.550422191619873, -6.305320739746094, 9.363828659057617, 0.09604430198669434
  };
  outerProduct(a, b, y, 10, 5, 32);
  for(int i = 0; i < 35; i++) {
    if(abs(c[i] - y[i]) > 1e-3) {
      passed = 0;
      std::cout << "outer: x differs from z by " << c[i] << " - " << y[i] << " = " << c[i] - y[i] << " at position " << i << std::endl;
    }
  }
  if(passed == 0)
    std::cerr << "Did not pass testOuterProduct for outerProduct." << std::endl;

  memset(y, 0, 10 * 5);
  outerProductAvx(a, b, y, 10, 5, 32);
  for(int i = 0; i < 35; i++) {
    if(abs(c[i] - y[i]) > 1e-3) {
      passed_avx = 0;
      std::cout << "outeravx: x differs from z by " << c[i] << " - " << y[i] << " = " << c[i] - y[i] << " at position " << i << std::endl;
    }
  }
  if(passed_avx == 0)
    std::cerr << "Did not pass testOuterProduct for outerProductAvx." << std::endl;
  if(passed_avx == 1 && passed == 1)
    std::cout << "Passed testOuterProduct" << std::endl;

}


float * gemmavxRelu(float * w, float * x, float * y, int bs, int dim_in, int dim_out) {
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       */
      __m512 sumx16 = _mm512_set1_ps(w[dim_in * dim_out + k] / 16);
      for(int j = 0; j < dim_in; j+=16) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m512 w_part = _mm512_loadu_ps(w + (k * dim_in + j));
        __m512 x_part = _mm512_loadu_ps(x + (i * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         */
        sumx16 = _mm512_fmadd_ps(w_part, x_part, sumx16);
      }
      y[i * dim_out + k] = (float)fmax(0, _mm512_reduce_add_ps(sumx16));
    }
  }
  return w + dim_in * dim_out + dim_out;
}


float * slicedGemm(float * w, const float * x, float * y, const int * indices, int bs, int dim_in, int dim_out, int spacing, int addBias) {
  // Implements a linear layer that computes x[idx, :]A + b for some rows of x.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of dim_out. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  int row_idx;
  for(int i = 0; i < bs; i++) {
    row_idx = indices[i];
    for(int k = 0; k < dim_out; k++) {
      y[i * spacing + k] = 0;
      for(int j = 0; j < dim_in; j++) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  row_idx = " << row_idx << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << row_idx * dim_in + j << "]";
        // std::cout << " | " << y[i * dim_out + k] << " += " << w[k * dim_in + j] << " * " << x[row_idx * dim_in + j] << std::endl;
        y[i * spacing + k] += w[k * dim_in + j] * x[row_idx * dim_in + j];
      }
    }
    if(addBias) {
      for (int j = 0; j < dim_out; j++) {
        y[i * spacing + j] += w[dim_in * dim_out + j];
      }
    }
  }
  return w + dim_in * dim_out + addBias * dim_out;
}


float * spacedOutputGemm(float * w, const float * x, float * y, int bs, int dim_in, int dim_out, int out_spacing) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  // The output for samples is stored with a spacing of dim_out. That is,
  // instead of storing the result of outputs sequentially, the output for
  // individual samples has a certain spacing.
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      y[i * out_spacing + k] = 0;
      for(int j = 0; j < dim_in; j++) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]";
        // std::cout << " | " << y[i * dim_out + k] << " += " << w[k * dim_in + j] << " * " << x[i * dim_in + j] << std::endl;
        y[i * out_spacing + k] += w[k * dim_in + j] * x[i * dim_in + j];
      }
    }
    for(int j = 0; j < dim_out; j++) {
      y[i * out_spacing + j] += w[dim_in * dim_out + j];
    }
  }
  return w + dim_in * dim_out + dim_out;
}


float * gemm(float * w, const float * x, float * y, int bs, int dim_in, int dim_out, int addBias) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      y[i * dim_out + k] = 0;
      for(int j = 0; j < dim_in; j++) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]";
        // std::cout << " | " << y[i * dim_out + k] << " += " << w[k * dim_in + j] << " * " << x[i * dim_in + j] << std::endl;
        y[i * dim_out + k] += w[k * dim_in + j] * x[i * dim_in + j];
      }
    }
    if(addBias) {
      for(int j = 0; j < dim_out; j++) {
        y[i * dim_out + j] += w[dim_in * dim_out + j];
      }
    }
  }
  return w + dim_in * dim_out + addBias * dim_out;
}


void relu(float * x, int length) {
  for(int i = 0; i < length; i++) {
    x[i] = ((float)(x[i] > 0)) * x[i];
  }
}


void argmax(const float * x, int * indices, int bs, int dim_in) {
  int maxIdx;
  float maxVal;
  int factori;
  float factorf;
  for(int i = 0; i < bs; i++) {
    maxIdx = 0;
    maxVal = -1e8;
    for(int j = 0; j < dim_in; j++) {
      factorf = (float)(x[i * dim_in + j] > maxVal);
      factori = (int)factorf;
      maxVal = factorf * x[i * dim_in + j] + (1 - factorf) * maxVal;
      maxIdx = factori * j + (1 - factori) * maxIdx;
    }
    indices[i] = maxIdx;
  }
}


void argmaxif(const float * x, int * indices, int bs, int dim_in) {
  int maxIdx;
  float maxVal;
  for(int i = 0; i < bs; i++) {
    maxIdx = 0;
    maxVal = -1000000000.;
    for(int j = 0; j < dim_in; j++) {
      if(x[i * dim_in + j] > maxVal) {
        maxVal = x[i * dim_in + j];
        maxIdx = j;
      };
    }
    indices[i] = maxIdx;
  }
}


void reluavx(float * x, int length) {
  __m512 zero = _mm512_set1_ps(0);
  for(int i = 0; i < length; i += 16) {
    __m512 part = _mm512_loadu_ps(x + i);
    part = _mm512_max_ps(part, zero);
    memcpy(x + i, (float*)&part, sizeof(float) * 16);
  }
}


void testLinear() {
  int dim_in = 32;
  int dim_out = 7;
  int bs = 5;
  auto y = (float*)calloc(sizeof(float), 35);
  float weights[] = {
          0.9048965573310852, 0.04869687557220459, 0.6072386503219604, 0.4452647566795349, 0.8454577326774597, 0.20122724771499634, 0.6304358839988708, 0.9850035905838013, 0.8317830562591553, 0.5830179452896118, 0.9839938282966614, 0.628241777420044, 0.5924873352050781, 0.1982085108757019, 0.6853355765342712, 0.7227343916893005, 0.13744789361953735, 0.4713612198829651, 0.15655934810638428, 0.13067024946212769, 0.4555739164352417, 0.6202710866928101, 0.3323821425437927, 0.13948369026184082, 0.5636759400367737, 0.6609397530555725, 0.9583190679550171, 0.7801330089569092, 0.6031395196914673, 0.5624909400939941, 0.6246362328529358, 0.11065101623535156,
          0.3507254719734192, 0.9464453458786011, 0.9501303434371948, 0.28168582916259766, 0.2801874876022339, 0.8939840793609619, 0.2935941815376282, 0.6994467973709106, 0.68051677942276, 0.6507982611656189, 0.673028290271759, 0.4833582043647766, 0.041475534439086914, 0.8186535239219666, 0.30551743507385254, 0.49036645889282227, 0.6618331074714661, 0.11377596855163574, 0.37946975231170654, 0.7027586102485657, 0.07531410455703735, 0.12810492515563965, 0.7425045967102051, 0.14549869298934937, 0.7560917139053345, 0.10692352056503296, 0.633652925491333, 0.42326533794403076, 0.5075516700744629, 0.8709064722061157, 0.5528688430786133, 0.25748682022094727,
          0.2896255850791931, 0.06652992963790894, 0.46668481826782227, 0.8407209515571594, 0.97240149974823, 0.2846173644065857, 0.8375775814056396, 0.21148574352264404, 0.9852117896080017, 0.3205645680427551, 0.05719304084777832, 0.20487570762634277, 0.8900133371353149, 0.9777494072914124, 0.342179536819458, 0.29819273948669434, 0.8381925821304321, 0.45766985416412354, 0.3413800001144409, 0.041815876960754395, 0.6720981597900391, 0.48132598400115967, 0.7582327723503113, 0.5555891990661621, 0.9681714773178101, 0.6698088645935059, 0.44688189029693604, 0.20871329307556152, 0.7464765906333923, 0.7141740918159485, 0.5627132058143616, 0.1811288595199585,
          0.7415596842765808, 0.17215222120285034, 0.3467937111854553, 0.6803995370864868, 0.2620311975479126, 0.015381693840026855, 0.3485926389694214, 0.3006158471107483, 0.40377968549728394, 0.5305456519126892, 0.7789401412010193, 0.6887916922569275, 0.009247303009033203, 0.3810872435569763, 0.13621681928634644, 0.8301608562469482, 0.9730076193809509, 0.2490224838256836, 0.44417309761047363, 0.7428154349327087, 0.04776829481124878, 0.17659014463424683, 0.09140318632125854, 0.8698099255561829, 0.9251696467399597, 0.5514410138130188, 0.6666351556777954, 0.7232200503349304, 0.5681109428405762, 0.6532554030418396, 0.9745791554450989, 0.8025661110877991,
          0.945043683052063, 0.40188294649124146, 0.8767057657241821, 0.49440211057662964, 0.22474753856658936, 0.3549983501434326, 0.3152781128883362, 0.37935054302215576, 0.1436532735824585, 0.3236147165298462, 0.1395389437675476, 0.6918264627456665, 0.6915835738182068, 0.02274423837661743, 0.7763278484344482, 0.7951418161392212, 0.6756926774978638, 0.7852311134338379, 0.6388113498687744, 0.6377953886985779, 0.9928251504898071, 0.6214389204978943, 0.9985864758491516, 0.05033397674560547, 0.9355177879333496, 0.4808627963066101, 0.2893807888031006, 0.08716398477554321, 0.4271628260612488, 0.524290919303894, 0.10008358955383301, 0.48679137229919434,
          0.08095341920852661, 0.24875915050506592, 0.7862083911895752, 0.20795708894729614, 0.34643054008483887, 0.6794365644454956, 0.16411876678466797, 0.8776107430458069, 0.5026704668998718, 0.0005664229393005371, 0.3304958939552307, 0.20584434270858765, 0.7008795142173767, 0.38203710317611694, 0.5789000391960144, 0.641463577747345, 0.011278688907623291, 0.8389354348182678, 0.8166813850402832, 0.6704944968223572, 0.1219586730003357, 0.020500123500823975, 0.033592402935028076, 0.935505747795105, 0.3033730983734131, 0.38655775785446167, 0.2606848478317261, 0.18429285287857056, 0.9873782992362976, 0.6463346481323242, 0.7369374632835388, 0.2465825080871582,
          0.8036155700683594, 0.3906424641609192, 0.1285858154296875, 0.8142405152320862, 0.25946271419525146, 0.6187949776649475, 0.8446365594863892, 0.9144771695137024, 0.968708872795105, 0.6007768511772156, 0.7328481674194336, 0.6679568886756897, 0.41181695461273193, 0.4705885648727417, 0.33308541774749756, 0.8691050410270691, 0.3934851884841919, 0.3109745383262634, 0.8512890934944153, 0.644864022731781, 0.07353931665420532, 0.5981866717338562, 0.9586911797523499, 0.596173107624054, 0.6514771580696106, 0.2606901526451111, 0.5357218980789185, 0.5738101005554199, 0.477902352809906, 0.7497110962867737, 0.3700951933860779, 0.615208625793457,

          -2.495177984237671, -1.9491448402404785, -1.8273398876190186, -0.7816107869148254, 2.3756372928619385, 0.3416927754878998, 0.661503791809082
  };
  float input[] = {
          0.6999265551567078, 0.47949594259262085, 0.7407759428024292, 0.9435071349143982, 0.14221221208572388, 0.08251065015792847, 0.35964930057525635, 0.34063011407852173, 0.6792567372322083, 0.664984941482544, 0.41472327709198, 0.5950785875320435, 0.7657796740531921, 0.022235333919525146, 0.6898019909858704, 0.1271255612373352, 0.6180364489555359, 0.4831085205078125, 0.43672478199005127, 0.6757811307907104, 0.7084929943084717, 0.6451724767684937, 0.7628821730613708, 0.3645080327987671, 0.9650511145591736, 0.24549728631973267, 0.1438104510307312, 0.47597241401672363, 0.08582335710525513, 0.5977247357368469, 0.8138248920440674, 0.09964466094970703,
          0.03921699523925781, 0.3273404836654663, 0.049195945262908936, 0.3667788505554199, 0.16433173418045044, 0.7187121510505676, 0.5628378391265869, 0.39776164293289185, 0.6415815949440002, 0.6888298392295837, 0.6656584739685059, 0.1909237504005432, 0.2477852702140808, 0.14832812547683716, 0.9216355085372925, 0.10513681173324585, 0.8131429553031921, 0.26339632272720337, 0.4388347268104553, 0.26211047172546387, 0.3620327115058899, 0.6759302020072937, 0.2933812737464905, 0.6073044538497925, 0.3425900936126709, 0.5011740922927856, 0.2585207223892212, 0.9864882826805115, 0.30577296018600464, 0.35303568840026855, 0.6261284947395325, 0.19793862104415894,
          0.5736050605773926, 0.7362657785415649, 0.9072589874267578, 0.5038405656814575, 0.917697012424469, 0.5710965991020203, 0.9523134231567383, 0.19996297359466553, 0.9374373555183411, 0.24310052394866943, 0.23100799322128296, 0.2607297897338867, 0.2543632984161377, 0.5340925455093384, 0.46762359142303467, 0.860028862953186, 0.6820789575576782, 0.6113433837890625, 0.47132980823516846, 0.8627652525901794, 0.43573594093322754, 0.6671746373176575, 0.16187715530395508, 0.7977359890937805, 0.38033849000930786, 0.06944626569747925, 0.5702359080314636, 0.36081862449645996, 0.8501364588737488, 0.3227250576019287, 0.014138936996459961, 0.040020763874053955,
          0.1607474684715271, 0.07336193323135376, 0.3494454026222229, 0.08480119705200195, 0.3935158848762512, 0.8493272066116333, 0.4277288317680359, 0.551157534122467, 0.2029774785041809, 0.748585045337677, 0.5410909652709961, 0.4989643096923828, 0.2921527028083801, 0.24922257661819458, 0.4166334271430969, 0.1988738775253296, 0.5888904333114624, 0.5067324638366699, 0.03394979238510132, 0.6958875060081482, 0.46438223123550415, 0.6865761280059814, 0.3104032278060913, 0.6426008343696594, 0.7768548727035522, 0.4287152886390686, 0.31818729639053345, 0.14501649141311646, 0.5753782391548157, 0.6421753168106079, 0.7511371970176697, 0.20398026704788208,
          0.8954839706420898, 0.1502983570098877, 0.5944675207138062, 0.14564919471740723, 0.8265166878700256, 0.17675191164016724, 0.19443464279174805, 0.9640757441520691, 0.15989232063293457, 0.7720884680747986, 0.4747067093849182, 0.9673083424568176, 0.6842890381813049, 0.7327936887741089, 0.8945381045341492, 0.04904717206954956, 0.37636375427246094, 0.4994449019432068, 0.8544185757637024, 0.4211924076080322, 0.950090765953064, 0.7956188917160034, 0.4958328604698181, 0.46064627170562744, 0.4263073801994324, 0.9791983366012573, 0.4834190011024475, 0.41659069061279297, 0.8158499598503113, 0.5925270915031433, 0.9910151958465576, 0.9373205900192261
  };
  float z[] = {
          6.10737419128418, 5.910383701324463, 6.799995422363281, 7.293015956878662, 11.574033737182617, 6.683558940887451, 9.885210037231445,
          4.816648483276367, 4.639202117919922, 5.048158645629883, 6.077128887176514, 8.406821250915527, 5.9289093017578125, 8.514716148376465,
          6.175973892211914, 6.430400371551514, 7.216405868530273, 7.204245090484619, 10.684303283691406, 7.961339950561523, 10.107136726379395,
          4.908487319946289, 5.196361064910889, 5.397024154663086, 6.400054454803467, 9.306540489196777, 6.691799640655518, 8.375603675842285,
          8.267187118530273, 6.721827983856201, 7.977182388305664, 8.690258026123047, 12.46012020111084, 8.823162078857422, 10.991238594055176
  };

  gemm(weights, input, y, bs, dim_in, dim_out, 1);
  for(int i = 0; i < 35; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "gemm: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }

  memset(y, 0, dim_out * bs);
  gemmavx(weights, input, y, bs, dim_in, dim_in, dim_out);
  for(int i = 0; i < 35; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "gemmavx: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }
}


void testSlicedGemm() {
  int dim_in = 32;
  int dim_out = 7;
  int spacing = 3;
  int bs = 2;
  int indices[] = {1, 3};
  auto y = (float*)calloc(sizeof(float), bs * dim_out * spacing);
  float weights[] = {
          0.9048965573310852, 0.04869687557220459, 0.6072386503219604, 0.4452647566795349, 0.8454577326774597, 0.20122724771499634, 0.6304358839988708, 0.9850035905838013, 0.8317830562591553, 0.5830179452896118, 0.9839938282966614, 0.628241777420044, 0.5924873352050781, 0.1982085108757019, 0.6853355765342712, 0.7227343916893005, 0.13744789361953735, 0.4713612198829651, 0.15655934810638428, 0.13067024946212769, 0.4555739164352417, 0.6202710866928101, 0.3323821425437927, 0.13948369026184082, 0.5636759400367737, 0.6609397530555725, 0.9583190679550171, 0.7801330089569092, 0.6031395196914673, 0.5624909400939941, 0.6246362328529358, 0.11065101623535156,
          0.3507254719734192, 0.9464453458786011, 0.9501303434371948, 0.28168582916259766, 0.2801874876022339, 0.8939840793609619, 0.2935941815376282, 0.6994467973709106, 0.68051677942276, 0.6507982611656189, 0.673028290271759, 0.4833582043647766, 0.041475534439086914, 0.8186535239219666, 0.30551743507385254, 0.49036645889282227, 0.6618331074714661, 0.11377596855163574, 0.37946975231170654, 0.7027586102485657, 0.07531410455703735, 0.12810492515563965, 0.7425045967102051, 0.14549869298934937, 0.7560917139053345, 0.10692352056503296, 0.633652925491333, 0.42326533794403076, 0.5075516700744629, 0.8709064722061157, 0.5528688430786133, 0.25748682022094727,
          0.2896255850791931, 0.06652992963790894, 0.46668481826782227, 0.8407209515571594, 0.97240149974823, 0.2846173644065857, 0.8375775814056396, 0.21148574352264404, 0.9852117896080017, 0.3205645680427551, 0.05719304084777832, 0.20487570762634277, 0.8900133371353149, 0.9777494072914124, 0.342179536819458, 0.29819273948669434, 0.8381925821304321, 0.45766985416412354, 0.3413800001144409, 0.041815876960754395, 0.6720981597900391, 0.48132598400115967, 0.7582327723503113, 0.5555891990661621, 0.9681714773178101, 0.6698088645935059, 0.44688189029693604, 0.20871329307556152, 0.7464765906333923, 0.7141740918159485, 0.5627132058143616, 0.1811288595199585,
          0.7415596842765808, 0.17215222120285034, 0.3467937111854553, 0.6803995370864868, 0.2620311975479126, 0.015381693840026855, 0.3485926389694214, 0.3006158471107483, 0.40377968549728394, 0.5305456519126892, 0.7789401412010193, 0.6887916922569275, 0.009247303009033203, 0.3810872435569763, 0.13621681928634644, 0.8301608562469482, 0.9730076193809509, 0.2490224838256836, 0.44417309761047363, 0.7428154349327087, 0.04776829481124878, 0.17659014463424683, 0.09140318632125854, 0.8698099255561829, 0.9251696467399597, 0.5514410138130188, 0.6666351556777954, 0.7232200503349304, 0.5681109428405762, 0.6532554030418396, 0.9745791554450989, 0.8025661110877991,
          0.945043683052063, 0.40188294649124146, 0.8767057657241821, 0.49440211057662964, 0.22474753856658936, 0.3549983501434326, 0.3152781128883362, 0.37935054302215576, 0.1436532735824585, 0.3236147165298462, 0.1395389437675476, 0.6918264627456665, 0.6915835738182068, 0.02274423837661743, 0.7763278484344482, 0.7951418161392212, 0.6756926774978638, 0.7852311134338379, 0.6388113498687744, 0.6377953886985779, 0.9928251504898071, 0.6214389204978943, 0.9985864758491516, 0.05033397674560547, 0.9355177879333496, 0.4808627963066101, 0.2893807888031006, 0.08716398477554321, 0.4271628260612488, 0.524290919303894, 0.10008358955383301, 0.48679137229919434,
          0.08095341920852661, 0.24875915050506592, 0.7862083911895752, 0.20795708894729614, 0.34643054008483887, 0.6794365644454956, 0.16411876678466797, 0.8776107430458069, 0.5026704668998718, 0.0005664229393005371, 0.3304958939552307, 0.20584434270858765, 0.7008795142173767, 0.38203710317611694, 0.5789000391960144, 0.641463577747345, 0.011278688907623291, 0.8389354348182678, 0.8166813850402832, 0.6704944968223572, 0.1219586730003357, 0.020500123500823975, 0.033592402935028076, 0.935505747795105, 0.3033730983734131, 0.38655775785446167, 0.2606848478317261, 0.18429285287857056, 0.9873782992362976, 0.6463346481323242, 0.7369374632835388, 0.2465825080871582,
          0.8036155700683594, 0.3906424641609192, 0.1285858154296875, 0.8142405152320862, 0.25946271419525146, 0.6187949776649475, 0.8446365594863892, 0.9144771695137024, 0.968708872795105, 0.6007768511772156, 0.7328481674194336, 0.6679568886756897, 0.41181695461273193, 0.4705885648727417, 0.33308541774749756, 0.8691050410270691, 0.3934851884841919, 0.3109745383262634, 0.8512890934944153, 0.644864022731781, 0.07353931665420532, 0.5981866717338562, 0.9586911797523499, 0.596173107624054, 0.6514771580696106, 0.2606901526451111, 0.5357218980789185, 0.5738101005554199, 0.477902352809906, 0.7497110962867737, 0.3700951933860779, 0.615208625793457,

          -2.495177984237671, -1.9491448402404785, -1.8273398876190186, -0.7816107869148254, 2.3756372928619385, 0.3416927754878998, 0.661503791809082
  };
  float input[] = {
          0.6999265551567078, 0.47949594259262085, 0.7407759428024292, 0.9435071349143982, 0.14221221208572388, 0.08251065015792847, 0.35964930057525635, 0.34063011407852173, 0.6792567372322083, 0.664984941482544, 0.41472327709198, 0.5950785875320435, 0.7657796740531921, 0.022235333919525146, 0.6898019909858704, 0.1271255612373352, 0.6180364489555359, 0.4831085205078125, 0.43672478199005127, 0.6757811307907104, 0.7084929943084717, 0.6451724767684937, 0.7628821730613708, 0.3645080327987671, 0.9650511145591736, 0.24549728631973267, 0.1438104510307312, 0.47597241401672363, 0.08582335710525513, 0.5977247357368469, 0.8138248920440674, 0.09964466094970703,
          0.03921699523925781, 0.3273404836654663, 0.049195945262908936, 0.3667788505554199, 0.16433173418045044, 0.7187121510505676, 0.5628378391265869, 0.39776164293289185, 0.6415815949440002, 0.6888298392295837, 0.6656584739685059, 0.1909237504005432, 0.2477852702140808, 0.14832812547683716, 0.9216355085372925, 0.10513681173324585, 0.8131429553031921, 0.26339632272720337, 0.4388347268104553, 0.26211047172546387, 0.3620327115058899, 0.6759302020072937, 0.2933812737464905, 0.6073044538497925, 0.3425900936126709, 0.5011740922927856, 0.2585207223892212, 0.9864882826805115, 0.30577296018600464, 0.35303568840026855, 0.6261284947395325, 0.19793862104415894,
          0.5736050605773926, 0.7362657785415649, 0.9072589874267578, 0.5038405656814575, 0.917697012424469, 0.5710965991020203, 0.9523134231567383, 0.19996297359466553, 0.9374373555183411, 0.24310052394866943, 0.23100799322128296, 0.2607297897338867, 0.2543632984161377, 0.5340925455093384, 0.46762359142303467, 0.860028862953186, 0.6820789575576782, 0.6113433837890625, 0.47132980823516846, 0.8627652525901794, 0.43573594093322754, 0.6671746373176575, 0.16187715530395508, 0.7977359890937805, 0.38033849000930786, 0.06944626569747925, 0.5702359080314636, 0.36081862449645996, 0.8501364588737488, 0.3227250576019287, 0.014138936996459961, 0.040020763874053955,
          0.1607474684715271, 0.07336193323135376, 0.3494454026222229, 0.08480119705200195, 0.3935158848762512, 0.8493272066116333, 0.4277288317680359, 0.551157534122467, 0.2029774785041809, 0.748585045337677, 0.5410909652709961, 0.4989643096923828, 0.2921527028083801, 0.24922257661819458, 0.4166334271430969, 0.1988738775253296, 0.5888904333114624, 0.5067324638366699, 0.03394979238510132, 0.6958875060081482, 0.46438223123550415, 0.6865761280059814, 0.3104032278060913, 0.6426008343696594, 0.7768548727035522, 0.4287152886390686, 0.31818729639053345, 0.14501649141311646, 0.5753782391548157, 0.6421753168106079, 0.7511371970176697, 0.20398026704788208,
          0.8954839706420898, 0.1502983570098877, 0.5944675207138062, 0.14564919471740723, 0.8265166878700256, 0.17675191164016724, 0.19443464279174805, 0.9640757441520691, 0.15989232063293457, 0.7720884680747986, 0.4747067093849182, 0.9673083424568176, 0.6842890381813049, 0.7327936887741089, 0.8945381045341492, 0.04904717206954956, 0.37636375427246094, 0.4994449019432068, 0.8544185757637024, 0.4211924076080322, 0.950090765953064, 0.7956188917160034, 0.4958328604698181, 0.46064627170562744, 0.4263073801994324, 0.9791983366012573, 0.4834190011024475, 0.41659069061279297, 0.8158499598503113, 0.5925270915031433, 0.9910151958465576, 0.9373205900192261
  };
  float z[] = {
          4.816648483276367, 4.639202117919922, 5.048158645629883, 6.077128887176514, 8.406821250915527, 5.9289093017578125, 8.514716148376465,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          4.908487319946289, 5.196361064910889, 5.397024154663086, 6.400054454803467, 9.306540489196777, 6.691799640655518, 8.375603675842285,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0
  };
  std::cout << "Execute SlicedGemm " << indices[0] << " " << indices[1] << std::endl;
  slicedGemm(weights, input, y, indices, bs, dim_in, dim_out, dim_out * spacing, 1);
  for(int i = 0; i < dim_out * bs * spacing; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "slicedGemm: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }

  std::cout << "Execute SlicedGemmAvx" << std::endl;
  memset(y, 0, dim_out * bs * spacing);
  slicedGemmavx(weights, input, y, indices, bs, dim_in, dim_out, dim_out * spacing);
  for(int i = 0; i < dim_out * bs * spacing; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "slicedGemmavx: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }
  free(y);
}


void testSpacedOutputGemm() {
  int dim_in = 32;
  int dim_out = 7;
  int bs = 3;
  int spacing = 3;
  auto y = (float*)calloc(sizeof(float), spacing * bs * dim_out);
  float weights[] = {
          0.9048965573310852, 0.04869687557220459, 0.6072386503219604, 0.4452647566795349, 0.8454577326774597, 0.20122724771499634, 0.6304358839988708, 0.9850035905838013, 0.8317830562591553, 0.5830179452896118, 0.9839938282966614, 0.628241777420044, 0.5924873352050781, 0.1982085108757019, 0.6853355765342712, 0.7227343916893005, 0.13744789361953735, 0.4713612198829651, 0.15655934810638428, 0.13067024946212769, 0.4555739164352417, 0.6202710866928101, 0.3323821425437927, 0.13948369026184082, 0.5636759400367737, 0.6609397530555725, 0.9583190679550171, 0.7801330089569092, 0.6031395196914673, 0.5624909400939941, 0.6246362328529358, 0.11065101623535156,
          0.3507254719734192, 0.9464453458786011, 0.9501303434371948, 0.28168582916259766, 0.2801874876022339, 0.8939840793609619, 0.2935941815376282, 0.6994467973709106, 0.68051677942276, 0.6507982611656189, 0.673028290271759, 0.4833582043647766, 0.041475534439086914, 0.8186535239219666, 0.30551743507385254, 0.49036645889282227, 0.6618331074714661, 0.11377596855163574, 0.37946975231170654, 0.7027586102485657, 0.07531410455703735, 0.12810492515563965, 0.7425045967102051, 0.14549869298934937, 0.7560917139053345, 0.10692352056503296, 0.633652925491333, 0.42326533794403076, 0.5075516700744629, 0.8709064722061157, 0.5528688430786133, 0.25748682022094727,
          0.2896255850791931, 0.06652992963790894, 0.46668481826782227, 0.8407209515571594, 0.97240149974823, 0.2846173644065857, 0.8375775814056396, 0.21148574352264404, 0.9852117896080017, 0.3205645680427551, 0.05719304084777832, 0.20487570762634277, 0.8900133371353149, 0.9777494072914124, 0.342179536819458, 0.29819273948669434, 0.8381925821304321, 0.45766985416412354, 0.3413800001144409, 0.041815876960754395, 0.6720981597900391, 0.48132598400115967, 0.7582327723503113, 0.5555891990661621, 0.9681714773178101, 0.6698088645935059, 0.44688189029693604, 0.20871329307556152, 0.7464765906333923, 0.7141740918159485, 0.5627132058143616, 0.1811288595199585,
          0.7415596842765808, 0.17215222120285034, 0.3467937111854553, 0.6803995370864868, 0.2620311975479126, 0.015381693840026855, 0.3485926389694214, 0.3006158471107483, 0.40377968549728394, 0.5305456519126892, 0.7789401412010193, 0.6887916922569275, 0.009247303009033203, 0.3810872435569763, 0.13621681928634644, 0.8301608562469482, 0.9730076193809509, 0.2490224838256836, 0.44417309761047363, 0.7428154349327087, 0.04776829481124878, 0.17659014463424683, 0.09140318632125854, 0.8698099255561829, 0.9251696467399597, 0.5514410138130188, 0.6666351556777954, 0.7232200503349304, 0.5681109428405762, 0.6532554030418396, 0.9745791554450989, 0.8025661110877991,
          0.945043683052063, 0.40188294649124146, 0.8767057657241821, 0.49440211057662964, 0.22474753856658936, 0.3549983501434326, 0.3152781128883362, 0.37935054302215576, 0.1436532735824585, 0.3236147165298462, 0.1395389437675476, 0.6918264627456665, 0.6915835738182068, 0.02274423837661743, 0.7763278484344482, 0.7951418161392212, 0.6756926774978638, 0.7852311134338379, 0.6388113498687744, 0.6377953886985779, 0.9928251504898071, 0.6214389204978943, 0.9985864758491516, 0.05033397674560547, 0.9355177879333496, 0.4808627963066101, 0.2893807888031006, 0.08716398477554321, 0.4271628260612488, 0.524290919303894, 0.10008358955383301, 0.48679137229919434,
          0.08095341920852661, 0.24875915050506592, 0.7862083911895752, 0.20795708894729614, 0.34643054008483887, 0.6794365644454956, 0.16411876678466797, 0.8776107430458069, 0.5026704668998718, 0.0005664229393005371, 0.3304958939552307, 0.20584434270858765, 0.7008795142173767, 0.38203710317611694, 0.5789000391960144, 0.641463577747345, 0.011278688907623291, 0.8389354348182678, 0.8166813850402832, 0.6704944968223572, 0.1219586730003357, 0.020500123500823975, 0.033592402935028076, 0.935505747795105, 0.3033730983734131, 0.38655775785446167, 0.2606848478317261, 0.18429285287857056, 0.9873782992362976, 0.6463346481323242, 0.7369374632835388, 0.2465825080871582,
          0.8036155700683594, 0.3906424641609192, 0.1285858154296875, 0.8142405152320862, 0.25946271419525146, 0.6187949776649475, 0.8446365594863892, 0.9144771695137024, 0.968708872795105, 0.6007768511772156, 0.7328481674194336, 0.6679568886756897, 0.41181695461273193, 0.4705885648727417, 0.33308541774749756, 0.8691050410270691, 0.3934851884841919, 0.3109745383262634, 0.8512890934944153, 0.644864022731781, 0.07353931665420532, 0.5981866717338562, 0.9586911797523499, 0.596173107624054, 0.6514771580696106, 0.2606901526451111, 0.5357218980789185, 0.5738101005554199, 0.477902352809906, 0.7497110962867737, 0.3700951933860779, 0.615208625793457,

          -2.495177984237671, -1.9491448402404785, -1.8273398876190186, -0.7816107869148254, 2.3756372928619385, 0.3416927754878998, 0.661503791809082
  };
  float input[] = {
          0.6999265551567078, 0.47949594259262085, 0.7407759428024292, 0.9435071349143982, 0.14221221208572388, 0.08251065015792847, 0.35964930057525635, 0.34063011407852173, 0.6792567372322083, 0.664984941482544, 0.41472327709198, 0.5950785875320435, 0.7657796740531921, 0.022235333919525146, 0.6898019909858704, 0.1271255612373352, 0.6180364489555359, 0.4831085205078125, 0.43672478199005127, 0.6757811307907104, 0.7084929943084717, 0.6451724767684937, 0.7628821730613708, 0.3645080327987671, 0.9650511145591736, 0.24549728631973267, 0.1438104510307312, 0.47597241401672363, 0.08582335710525513, 0.5977247357368469, 0.8138248920440674, 0.09964466094970703,
          0.03921699523925781, 0.3273404836654663, 0.049195945262908936, 0.3667788505554199, 0.16433173418045044, 0.7187121510505676, 0.5628378391265869, 0.39776164293289185, 0.6415815949440002, 0.6888298392295837, 0.6656584739685059, 0.1909237504005432, 0.2477852702140808, 0.14832812547683716, 0.9216355085372925, 0.10513681173324585, 0.8131429553031921, 0.26339632272720337, 0.4388347268104553, 0.26211047172546387, 0.3620327115058899, 0.6759302020072937, 0.2933812737464905, 0.6073044538497925, 0.3425900936126709, 0.5011740922927856, 0.2585207223892212, 0.9864882826805115, 0.30577296018600464, 0.35303568840026855, 0.6261284947395325, 0.19793862104415894,
          0.5736050605773926, 0.7362657785415649, 0.9072589874267578, 0.5038405656814575, 0.917697012424469, 0.5710965991020203, 0.9523134231567383, 0.19996297359466553, 0.9374373555183411, 0.24310052394866943, 0.23100799322128296, 0.2607297897338867, 0.2543632984161377, 0.5340925455093384, 0.46762359142303467, 0.860028862953186, 0.6820789575576782, 0.6113433837890625, 0.47132980823516846, 0.8627652525901794, 0.43573594093322754, 0.6671746373176575, 0.16187715530395508, 0.7977359890937805, 0.38033849000930786, 0.06944626569747925, 0.5702359080314636, 0.36081862449645996, 0.8501364588737488, 0.3227250576019287, 0.014138936996459961, 0.040020763874053955,
  };
  float z[] = {
          6.10737419128418, 5.910383701324463, 6.799995422363281, 7.293015956878662, 11.574033737182617, 6.683558940887451, 9.885210037231445,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          4.816648483276367, 4.639202117919922, 5.048158645629883, 6.077128887176514, 8.406821250915527, 5.9289093017578125, 8.514716148376465,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          6.175973892211914, 6.430400371551514, 7.216405868530273, 7.204245090484619, 10.684303283691406, 7.961339950561523, 10.107136726379395,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0
  };

  spacedOutputGemm(weights, input, y, bs, dim_in, dim_out, dim_out * spacing);
  for(int i = 0; i < spacing * bs * dim_out; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "spacedOutputGemm: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }

  memset(y, 0, dim_out * bs * spacing);
  spacedOutputGemmavx(weights, input, y, bs, dim_in, dim_in, dim_out, dim_out * spacing);
  for(int i = 0; i < spacing * bs * dim_out; i++) {
    if(abs(z[i] - y[i]) > 1e-3) {
      std::cout << "spacedOutputGemmavx: x differs from z by " << z[i] << " - " << y[i] << " = " << z[i] - y[i] << " at position " << i << std::endl;
    }
  }
  free(y);
}


int measureGemm() {
  std::cout << std::endl <<  "Measure Gemm and Gemmavx" << std::endl << "==============================" << std::endl;
  int dim_in = 896;
  int dim_out = 128;
  int bs = 127;
  double avg_loop, avg_avx;
  // dim_in = 64;
  // dim_out = 32;
  // bs = 16;
  auto params = (float *)calloc((dim_in * dim_out + dim_out), sizeof(float));
  auto x      = (float *)calloc((bs * dim_in), sizeof(float));
  auto y      = (float *)calloc((bs * dim_out), sizeof(float));
  if (params == nullptr || x == nullptr || y == nullptr){
    std::cerr << "Memory not correctly initialized!" << std::endl;
    return -1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    gemmavx(params, x, y, bs, dim_in, dim_in, dim_out);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    gemm(params, x, y, bs, dim_in, dim_out, 1);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "Execution took " << avg_loop * 1000 << "ms" << std::endl;
  std::cout << "AVX512 is " << avg_loop / avg_avx << " times faster than looping" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


int measureSpacedOutputGemm() {
  std::cout << std::endl <<  "Measure SpacedOutputGemm and SpacedOutputGemmavx" << std::endl << "==============================" << std::endl;
  int dim_in = 896;
  int dim_out = 128;
  int bs = 127;
  int stride = 256;
  double avg_loop, avg_avx;
  // dim_in = 64;
  // dim_out = 32;
  // bs = 16;
  auto params = (float *)calloc((dim_in * dim_out + dim_out), sizeof(float));
  auto x      = (float *)calloc((bs * dim_in), sizeof(float));
  auto y      = (float *)calloc((bs * stride), sizeof(float));
  if (params == nullptr || x == nullptr || y == nullptr){
    std::cerr << "Memory not correctly initialized!" << std::endl;
    return -1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    spacedOutputGemmavx(params, x, y, bs, dim_in, dim_in, dim_out, stride);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "SpacedOutputGemmavx Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    spacedOutputGemm(params, x, y, bs, dim_in, dim_out, stride);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "SpacedOutputGemm Execution took " << avg_loop * 1000 << "ms" << std::endl;
  std::cout << "AVX512 is " << avg_loop / avg_avx << " times faster than looping" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


int measureOuterProduct() {
  int dim_a = 2048;
  int dim_b = 1024;
  int ncol = 128;

  double avg_loop, avg_avx;
  // dim_in = 64;
  // dim_out = 32;
  // bs = 16;
  auto a = (float *)calloc((dim_a * ncol), sizeof(float));
  auto b = (float *)calloc((dim_b * ncol), sizeof(float));
  auto y = (float *)calloc((dim_a * dim_b), sizeof(float));

  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    outerProductAvx(a, b, y, dim_a, dim_b, ncol);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    outerProduct(a, b, y, dim_a, dim_b, ncol);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "Execution took " << avg_loop * 1000 << "ms" << std::endl;

  std::cout << "AVX512 is " << avg_loop / avg_avx << " times faster than looping" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


int measureFusedGemm() {
  int dim_in = 2048;
  int dim_out = 1024;
  int bs = 32;
  double avg_loop, avg_avx;
  // dim_in = 64;
  // dim_out = 32;
  // bs = 16;
  auto params = (float *)calloc((dim_in * dim_out + dim_out), sizeof(float));
  auto x      = (float *)calloc((bs * dim_in), sizeof(float));
  auto y      = (float *)calloc((bs * dim_out), sizeof(float));
  if (params == nullptr || x == nullptr || y == nullptr){
    std::cerr << "Memory not correctly initialized!" << std::endl;
    return -1;
  }
  std::cout << x << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    gemmavxRelu(params, x, y, bs, dim_in, dim_out);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "Fused Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    gemmavx(params, x, y, bs, dim_in, dim_in, dim_out);
    relu(y, bs * dim_out);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "Execution took " << avg_loop * 1000 << "ms" << std::endl;
  std::cout << "Fused is " << avg_loop / avg_avx << " times faster than non-fused" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


int testRelu() {
  int dim = 32;
  auto x = (float*)malloc(sizeof(float) * dim);
  auto z = (float*)malloc(sizeof(float) * dim);

  for(int i = 0; i < dim; i++) {
    if(i % 2 == 0) {
      x[i] = (float)(-1 * i);
      z[i] = 0;
    } else{
      x[i] = (float)i;
      z[i] = (float)i;
    }
  }
  reluavx(x, dim);
  for(int i = 0; i < dim; i++) {
    if(z[i] != x[i]) {
      std::cout << "Target different from prediction at position " << i;
      std::cout << ": " << z[i] << " != " << x[i] << std::endl;
    }
  }

  for(int i = 0; i < dim; i++) {
    if(i % 2 == 0) {
      x[i] = (float)(-1 * i);
      z[i] = 0;
    } else{
      x[i] = (float)i;
      z[i] = (float)i;
    }
  }
  relu(x, dim);
  for(int i = 0; i < dim; i++) {
    if(z[i] != x[i]) {
      std::cout << "Target different from prediction at position " << i;
      std::cout << ": " << z[i] << " != " << x[i] << std::endl;
    }
  }
  std::cout << "Finished test relu" <<  std::endl;
  return 0;
}


int measureRelu() {
  int dim = 1024 * 1024;
  auto x = (float*)malloc(sizeof(float) * dim);

  double avg_loop, avg_avx;
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    reluavx(x, dim);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "AVX Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    relu(x, dim);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "Loop Execution took " << avg_loop * 1000 << "ms" << std::endl;
  std::cout << "AVX512 is " << avg_loop / avg_avx << " times faster than looping" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


int measureArgmax() {
  int dim = 1024 * 1024;
  auto x = (float*)malloc(sizeof(float) * dim);
  auto idx = (int*)malloc(sizeof(int) * 1024);

  double avg_loop, avg_avx;
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    argmax(x, idx, 1024, 1024);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  avg_avx = diff.count() / 100;
  std::cout << "Compute Execution took " << avg_avx * 1000 << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    argmaxif(x, idx, 1024, 1024);
  }
  stop = std::chrono::high_resolution_clock::now();
  diff = stop - start;
  avg_loop = diff.count() / 100;
  std::cout << "If Execution took " << avg_loop * 1000 << "ms" << std::endl;
  std::cout << "Compute is " << avg_loop / avg_avx << " times faster than if" << std::endl;
  std::cout << "finish" << std::endl;
  return 0;
}


void forwardAvx(struct NeuralNet * nn, int * predictions, float *buf1, float *buf2, int *intBuf) {
  int stride = 16;
  float * w1, w2, w3;
  int offset1, offset2;
  w1 = nn->nnParams;
  for(int head = 0; head < nn->config->hlsa_num_heads; head++) {
    /*
     * To avoid unnecssary computations while, under the AVX code
     * we need the spacedOutputGemmAvx function to compute the transformations.
     * The output of the transformations is the input to the outer product.
     * The outer product uses the output dimension of the two transformations
     * as input dimension. Thus, the function loads 16 elements of the output
     * at once. If we pack the output denseley, then the outer product loads
     * parts of the next row. This can result in wrong calculations.
     * To avoid this, we space the output by a multiple of 16. Since the
     * memory region is initialized with zeros, the outer product code
     * will only load zeros, multiply those together, and add zero to the
     * result, thus preserving the correct calculations.
     *
     * For other gemmavx calls this adjustment is not necessary. While the
     * calls will also read beyond the true input if the output dim of the
     * previous layer is not a multiple of 16, wrong calculations will not
     * occur since the weights are zero on those positions. Here, we only
     * have to adapt the input dimensions.
     */
    // Transform the keys into the latent space. Batch size is 80 in this
    // case for the number of switches that we have. Returned tensor has
    // shape of (numSwitches, hlsa_dim_hidden).
    if(LOGLEVEL > 0) {
      std::cout << std::endl << "Transform Keys for Head " << head + 1 << std::endl;
      std::cout << "spacedOutputGemmavxNoBias(w1, nn->allIps, buf1"
                << ", bs=" << nn->numSwitches
                << ", dim_in=" << nn->config->hlsa_dim_k_avx
                << ", dim_out=" << nn->config->hlsa_dim_hidden
                << ", stride=" << nn->config->hlsa_dim_hidden_avx << ")" << std::endl;
    }
    w1 = spacedOutputGemmavxNoBias(
            w1,
            nn->allIps,
            buf1,
            nn->numSwitches,
            nn->config->hlsa_dim_k,
            nn->config->hlsa_dim_k_avx,
            nn->config->hlsa_dim_hidden,
            nn->config->hlsa_dim_hidden_avx
    );
    // Transform queries into the latent space. Batch size correspond to the
    // number of samples asked for. Returned tensor has shape of (numSamples, hlsa_dim_Hidden)
    offset1 = nn->numSwitches * nn->config->hlsa_dim_hidden_avx;
    if(LOGLEVEL > 0) {
      std::cout << std::endl << "Transform Queries for Head " << head + 1 << std::endl;
      std::cout << "spacedOutputGemmavxNoBias(w1, nn->queries, buf1 + " << offset1
                << ", bs=" << nn->numSamples
                << ", dim_in=" << nn->config->hlsa_dim_q_avx
                << ", dim_out=" << nn->config->hlsa_dim_hidden
                << ", stride=" << nn->config->hlsa_dim_hidden_avx << ")" << std::endl;
    }
    w1 = spacedOutputGemmavxNoBias(
            w1,
            nn->queries,
            buf1 + offset1,
            nn->numSamples,
            nn->config->hlsa_dim_q,
            nn->config->hlsa_dim_q_avx,
            nn->config->hlsa_dim_hidden,
            nn->config->hlsa_dim_hidden_avx
    );
    // Compute the outer product between the transformed keys and queries.
    // buf1 points to the row major (numSwitches, hlsa_dim_hidden) matrix.
    // buf1 + offset to a row major (numSamples, hlsa_dim_hidden) matrix.
    // Ther resulting column major result matrix of shape (numSwitches, numSamples)
    // is stored at offset2
    offset2 = offset1 + nn->numSamples * nn->config->hlsa_dim_hidden_avx;
    if(LOGLEVEL > 0) {
      std::cout << std::endl << "Calculate outer product for Head " << head + 1 << std::endl;
      std::cout << "outerProduct(buf1, buf1 + " << offset1 << ", buf2 + " << offset2
                << ", nrows_a=" << nn->numSwitches
                << ", nrows_b=" << nn->numSamples
                << ", ncols=" << nn->config->hlsa_dim_hidden_avx << ")" << std::endl;
    }
    outerProductAvx(
            buf1,
            buf1 + offset1,
            buf1 + offset2,
            nn->numSwitches,
            nn->numSamples,
            nn->config->hlsa_dim_hidden_avx
    );
    // Get the indices of the maximum values. Buf1 does now contain the unscaled
    // and unnormalized attention scores. Scaling and normalization changes
    // the scores values, but not the position of the maximum. Thus, do not
    // scale and normalize.
    if(LOGLEVEL > 0) {
      std::cout << "Compute argmax for Head " << head + 1 << std::endl;
      std::cout << "agmaxif(buf1 + " << offset2 << ", intBuf, "
                << nn->numSamples << ", " << nn->numSwitches << ")" << std::endl;
    }
    argmaxif(buf1 + offset2, intBuf, nn->numSamples, nn->numSwitches);
    // Retrieve the indices of the switches with the maximum outer product
    // value, use the indices to retrieve the corresponding HNSA, transform
    // the HNSA into a latent space and store the transformed HNSA at a
    // spacing to each other that preserves the sampes. That is, for each
    // sample, the attention heads are executed one after the other and
    // the results subsequently added to the output.
    if(LOGLEVEL > 0) {
      std::cout << std::endl << "Transform Values for Head " << head + 1 << std::endl;
      std::cout << "slicedGemmavx(w1, nn->hnsas, buf2, intBuf,"
                << ", bs=" << nn->numSamples
                << ", dim_in=" << nn->config->hlsa_dim_avx
                << ", dim_out=" << nn->config->hlsa_dim_out
                << ", stride=" << nn->config->hlsa_dim_out * nn->config->hlsa_dim_hidden_avx << ")" << std::endl;
    }
    w1 = slicedGemmavxNoBias(
            w1,
            nn->hnsas,
            buf2 + head * nn->config->hlsa_dim_out,
            intBuf,
            nn->numSamples,
            nn->config->hlsa_dim,
            nn->config->hlsa_dim_avx,
            nn->config->hlsa_dim_out,
            nn->config->hlsa_dim_out * nn->config->hlsa_num_heads
    );
    if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, 700);
  }
  // All Attention heads have been processed and the outputs of each head
  // are concatenated to each other. Now transform each of those with the
  // Linear transformation, and store them with some spacing inbetween. The
  // spacing will be filled with the transformation from the neighbor fcn.
  memset(buf1, 0, 2000);
  if(LOGLEVEL > 0) {
    std::cout << std::endl << "Calculate the attention module output from all heads." << std::endl;
    std::cout << "spacedOutputGemmAvx(w1, buf2, buf1"
              << ", bs=" << nn->numSamples
              << ", dim_in=" << nn->config->hlsa_dim_out_avx
              << ", dim_out=" << nn->config->hlsa_dim_fcn
              << ", stride=" << nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn << ");" << std::endl;
  }
  if (LOGLEVEL > 1) printColumnMajorMatrix(w1, nn->config->hlsa_dim_out_avx, nn->config->hlsa_dim_fcn);
  w1 = spacedOutputGemmavx(
          w1,
          buf2,
          buf1,
          nn->numSamples,
          nn->config->hlsa_num_heads * nn->config->hlsa_dim_out,
          nn->config->hlsa_dim_out_avx,
          nn->config->hlsa_dim_fcn,
          nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn
  );
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn);
  // Compute the transformation from the concatenation of destination and current
  // location IP. Offset y with the length of the hlsa fcn output dim. Store
  // the transoformation with the spacing inbetween.
  if(LOGLEVEL > 0) {
    std::cout << std::endl << "Encode neighbor and current location for location module." << std::endl;
    std::cout << "spacedOutputGemmAvx(w1, buf2, buf1 + " << nn->config->hlsa_dim_out
              << ", bs=" << nn->numSamples
              << ", dim_in=" << nn->config->hlsa_dim_q_avx
              << ", dim_out=" << nn->config->neighbor_fcn
              << ", stride=" << nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn << ");" << std::endl;
  }
  w1 = spacedOutputGemmavx(
          w1,
          nn->queries,
          buf1 + nn->config->hlsa_dim_fcn,
          nn->numSamples,
          nn->config->hlsa_dim_q,
          nn->config->hlsa_dim_q_avx,
          nn->config->neighbor_fcn,
          nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn
  );
  // Apply the relu once for the concatenated elements.
  relu(buf1, nn->numSamples * (nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn));
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn);
  // perform the first of the final output transformations.
  if(LOGLEVEL > 0) {
    std::cout << "Transform output from NSMod and LocMod." << std::endl;
    std::cout << "gemmavx(w1, buf1, buf2 + " << nn->config->hlsa_dim_out
              << ", bs=" << nn->numSamples
              << ", dim_in=" << nn->config->hlsa_out_neighbor_embd_avx
              << ", dim_out=" << nn->config->final_fcn_0  << ");" << std::endl;
  }
  w1 = gemmavx(
          w1,
          buf1,
          buf2,
          nn->numSamples,
          nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn,
          nn->config->hlsa_out_neighbor_embd_avx,
          nn->config->final_fcn_0
  );
  relu(buf2, nn->numSamples * nn->config->final_fcn_0);
  if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, nn->config->final_fcn_0);
  // perform the second of the final output transformations.
  if(LOGLEVEL > 0) {
    std::cout << std::endl << "Second hidden layer in out mod." << std::endl;
    std::cout << "gemmavx(w1, buf2, buf1"
              << ", bs=" << nn->numSamples
              << ", dim_in=" << nn->config->final_fcn_0_avx
              << ", dim_out=" << nn->config->final_fcn_1  << ");" << std::endl;
  }
  w1 = gemmavx(
          w1,
          buf2,
          buf1,
          nn->numSamples,
          nn->config->final_fcn_0,
          nn->config->final_fcn_0_avx,
          nn->config->final_fcn_1
  );
  relu(buf1, nn->numSamples * nn->config->final_fcn_1);
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->final_fcn_1);
  // Calculate the output logits.
  if(LOGLEVEL > 0) {
    std::cout << std::endl << "Output layer in out mod." << std::endl;
    std::cout << "gemmavx(w1, buf1, buf2"
              << ", bs=" << nn->numSamples
              << ", dim_in=" << nn->config->final_fcn_1_avx
              << ", dim_out=" << nn->config->dim_out << ");" << std::endl;
  }
  gemmavx(
          w1,
          buf1,
          buf2,
          nn->numSamples,
          nn->config->final_fcn_1,
          nn->config->final_fcn_1_avx,
          nn->config->dim_out
  );
  if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, nn->config->dim_out);
  // Calculate the output indices.
  if(LOGLEVEL > 0) {
    std::cout << std::endl << "Calculate output index." << std::endl;
    std::cout << "agmaxif(buf2, intBuf, "
              << nn->numSamples << ", " << nn->config->dim_out << ")" << std::endl;
  }
  argmaxif(buf2, predictions, nn->numSamples, nn->config->dim_out);
}


void *mpForwardAvx(void * targs) {
  struct thread_args * args;
  args = (struct thread_args *)targs;
  //std::cout << "Thread " << args->thread_id << " " << args->nn << " " << args->predictions << " " << args->buf1 << " " << args->buf2 << " " << args->intBuf << std::endl;
  forwardAvx(args->nn, args->predictions, args->buf1, args->buf2, args->intBuf);
  pthread_exit(nullptr);
}


void forward(struct NeuralNet * nn, int * predictions, float *buf1, float *buf2, int *intBuf) {
  float * w1, w2, w3;
  int offset1, offset2;
  w1 = nn->nnParams;
  for(int head = 0; head < nn->config->hlsa_num_heads; head++) {
    // Transform the keys into the latent space. Batch size is 80 in this
    // case for the number of switches that we have. Returned tensor has
    // shape of (numSwitches, hlsa_dim_hidden).
    if(LOGLEVEL > 0) { std::cout << "Transform Keys for Head " << head + 1 << std::endl; }
    w1 = gemm(
            w1,
            nn->allIps,
            buf1,
            nn->numSwitches,
            nn->config->hlsa_dim_k,
            nn->config->hlsa_dim_hidden,
            0
    );
    // Transform queries into the latent space. Batch size correspond to the
    // number of samples asked for. Returned tensor has shape of (numSamples, hlsa_dim_Hidden)
    offset1 = nn->numSwitches * nn->config->hlsa_dim_hidden;
    if(LOGLEVEL > 0) { std::cout << "Transform Queries for Head " << head + 1 << std::endl; }
    w1 = gemm(
            w1,
            nn->queries,
            buf1 + offset1,
            nn->numSamples,
            nn->config->hlsa_dim_q,
            nn->config->hlsa_dim_hidden,
            0
    );
    // Compute the outer product between the transformed keys and queries.
    // buf1 points to the row major (numSwitches, hlsa_dim_hidden) matrix.
    // buf1 + offset to a row major (numSamples, hlsa_dim_hidden) matrix.
    // Ther resulting column major result matrix of shape (numSwitches, numSamples)
    // is stored at offset2
    offset2 = offset1 + nn->numSamples * nn->config->hlsa_dim_hidden;
    if(LOGLEVEL > 0) { std::cout << "Calculate outer product for Head " << head + 1 << std::endl; }
    outerProduct(
            buf1,
            buf1 + offset1,
            buf1 + offset2,
            nn->numSwitches,
            nn->numSamples,
            nn->config->hlsa_dim_hidden
    );
    // Get the indices of the maximum values. Buf1 does now contain the unscaled
    // and unnormalized attention scores. Scaling and normalization changes
    // the scores values, but not the position of the maximum. Thus, do not
    // scale and normalize.
    if(LOGLEVEL > 0) { std::cout << "Compute argmax for Head " << head + 1 << std::endl; }
    argmaxif(buf1 + offset2, intBuf, nn->numSamples, nn->numSwitches);
    // Retrieve the indices of the switches with the maximum outer product
    // value, use the indices to retrieve the corresponding HNSA, transform
    // the HNSA into a latent space and store the transformed HNSA at a
    // spacing to each other that preserves the sampes. That is, for each
    // sample, the attention heads are executed one after the other and
    // the results subsequently added to the output.
    if(LOGLEVEL > 0) { std::cout << "Transform Values for Head " << head + 1 << std::endl; }
    w1 = slicedGemm(
            w1,
            nn->hnsas,
            buf2 + head * nn->config->hlsa_dim_out,
            intBuf,
            nn->numSamples,
            nn->config->hlsa_dim,
            nn->config->hlsa_dim_out,
            nn->config->hlsa_dim_out * nn->config->hlsa_num_heads,
            0
    );
    if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, 700);
  }
  // All Attention heads have been processed and the outputs of each head
  // are concatenated to each other. Now transform each of those with the
  // Linear transformation, and store them with some spacing inbetween. The
  // spacing will be filled with the transformation from the neighbor fcn.
  // memset(buf1, 0, 2000);
  if(LOGLEVEL > 0) { std::cout << "Calculate the attention module output from all heads." << std::endl;
    std::cout << "spacedOutputGemm(w1, buf2, buf1, " << nn->numSamples << ", "
              << nn->config->hlsa_num_heads * nn->config->hlsa_dim_out << ", "
              << nn->config->hlsa_dim_fcn << ", " << nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn << ");" << std::endl;
  }
  w1 = spacedOutputGemm(
          w1,
          buf2,
          buf1,
          nn->numSamples,
          nn->config->hlsa_num_heads * nn->config->hlsa_dim_out,
          nn->config->hlsa_dim_fcn,
          nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn
  );
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn);
  // Compute the transformation from the concatenation of destination and current
  // location IP. Offset y with the length of the hlsa fcn output dim. Store
  // the transoformation with the spacing inbetween.
  if(LOGLEVEL > 0) { std::cout << "Encode neighbor and current location for location module." << std::endl; }
  w1 = spacedOutputGemm(
          w1,
          nn->queries,
          buf1 + nn->config->hlsa_dim_fcn,
          nn->numSamples,
          2 * nn->config->dim_embedding,
          nn->config->neighbor_fcn,
          nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn);
  // Apply the relu once for the concatenated elements.
  relu(buf1, nn->numSamples * (nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn));
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->neighbor_fcn + nn->config->hlsa_dim_fcn);
  // perform the first of the final output transformations.
  if(LOGLEVEL > 0) { std::cout << "Transform output from NSMod and LocMod." << std::endl; }
  w1 = gemm(
          w1,
          buf1,
          buf2,
          nn->numSamples,
          nn->config->hlsa_dim_fcn + nn->config->neighbor_fcn,
          nn->config->final_fcn_0,
          1
  );
  relu(buf2, nn->numSamples * nn->config->final_fcn_0);
  if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, nn->config->final_fcn_0);
  // perform the second of the final output transformations.
  if(LOGLEVEL > 0) { std::cout << "Second hidden layer in out mod." << std::endl; }
  w1 = gemm(
          w1,
          buf2,
          buf1,
          nn->numSamples,
          nn->config->final_fcn_0,
          nn->config->final_fcn_1,
          1
  );
  relu(buf1, nn->numSamples * nn->config->final_fcn_1);
  if (LOGLEVEL > 1) printRowMajorMatrix(buf1, 2, nn->config->final_fcn_1);
  // Calculate the output logits.
  if(LOGLEVEL > 0) { std::cout << "Output layer in out mod." << std::endl; }
  gemm(
          w1,
          buf1,
          buf2,
          nn->numSamples,
          nn->config->final_fcn_1,
          nn->config->dim_out,
          1
  );
  if (LOGLEVEL > 1) printRowMajorMatrix(buf2, 2, nn->config->dim_out);
  // Calculate the output indices.
  if(LOGLEVEL > 0) { std::cout << "Calculate output index." << std::endl; }
  argmaxif(buf2, predictions, nn->numSamples, nn->config->dim_out);
}


void measureForwardAvx() {
  std::cout << std::endl <<  "Measure ForwardAvx" << std::endl << "==============================" << std::endl;
  struct NnConfig config;
  // adaptNnConfigToAvx(&config, 16);
  struct NeuralNet nn;
  nn.numSamples = 127;
  nn.numSwitches = 80;
  nn.config = &config;

  auto predictions = (int*)calloc(127     , sizeof(int));
  auto intBuf      = (int*)calloc(127 * 10, sizeof(int));

  auto buf1        = (float*)calloc(127 * 1000, sizeof(float));
  auto buf2        = (float*)calloc(127 * 1000, sizeof(float));
  nn.hnsas         = (float*)calloc(127 * 128 , sizeof(float));
  nn.allIps        = (float*)calloc(127 * 32  , sizeof(float));
  nn.dstIps        = (float*)calloc(127 * 32  , sizeof(float));
  nn.switchIps     = (float*)calloc(127 * 32  , sizeof(float));
  nn.queries       = (float*)calloc(127 * 64  , sizeof(float));
  nn.nnParams      = (float*)calloc(300000    , sizeof(float));

  fillAllIps(nn.allIps);

  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    nn.numSamples = 123;
    forwardAvx(&nn, predictions, buf1, buf2, intBuf);
    nn.numSamples = 112;
    forwardAvx(&nn, predictions, buf1, buf2, intBuf);
  }
  auto stop  = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  double avg = diff.count() / 100;
  std::cout << "Execution took on average " << avg * 1000 << "ms" << std::endl;

  free(predictions);
  free(intBuf);
  free(buf1);
  free(buf2);
  free(nn.hnsas);
  free(nn.allIps);
  free(nn.dstIps);
  free(nn.switchIps);
  free(nn.queries);
  free(nn.nnParams);
}

/*
void measureMpForwardAvx() {
  std::cout << std::endl <<  "Measure mpForwardAvx" << std::endl << "==============================" << std::endl;
  struct NnConfig config;
  // adaptNnConfigToAvx(&config, 16);
  struct NeuralNet threadNns[NUM_THREADS];
  struct thread_args args[NUM_THREADS];
  pthread_t threads[NUM_THREADS];

  struct NeuralNet nn;
  nn.numSamples = 127;
  nn.numSwitches = 80;
  nn.config = &config;

  auto predictions = (int*)calloc(NUM_THREADS * 127     , sizeof(int));
  auto intBuf      = (int*)calloc(NUM_THREADS * 127 * 10, sizeof(int));

  auto buf1        = (float*)calloc(NUM_THREADS * 127 * 1000, sizeof(float));
  auto buf2        = (float*)calloc(NUM_THREADS * 127 * 1000, sizeof(float));
  nn.hnsas         = (float*)calloc(127 * 128 , sizeof(float));
  nn.allIps        = (float*)calloc(127 * 32  , sizeof(float));
  nn.nnParams      = (float*)calloc(300000    , sizeof(float));
  nn.dstIps        = (float*)calloc(127 * 32  , sizeof(float));
  nn.switchIps     = (float*)calloc(127 * 32  , sizeof(float));
  nn.queries       = (float*)calloc(127 * 64  , sizeof(float));

  fillAllIps(nn.allIps);

  for(auto & threadNn : threadNns) {
    threadNn.numSwitches = nn.numSwitches;
    threadNn.config = nn.config;
    threadNn.hnsas = nn.hnsas;
    threadNn.allIps = nn.allIps;
    threadNn.switchIps = nn.switchIps;
    threadNn.nnParams = nn.nnParams;
  }
  for(int i = 0; i < NUM_THREADS; i++) {
    args[i].thread_id = i;
    args[i].intBuf = intBuf + i * 127 * 10;
    args[i].predictions = predictions + i * 127;
    args[i].buf1 = buf1 + i * 127 * 1000;
    args[i].buf2 = buf2 + i * 127 * 1000;
    args[i].nn = &threadNns[i];
    std::cout << "Thread " << args[i].thread_id << " " << args[i].nn << " " << args[i].predictions << " "
    << args[i].buf1 << " " << args[i].buf2 << " " << args[i].intBuf << std::endl;

  }

  int rc;
  void *status;
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    nn.numSamples = 123;
    int wholes = nn.numSamples / NUM_THREADS;
    int remainder = nn.numSamples % NUM_THREADS;
    int idx = 0;
    for(int j = 0; j < NUM_THREADS; j++) {
      threadNns[j].numSamples = wholes + (int)(remainder > 0);
      // std::cout << "Start thread " << j << " " << idx << " " << remainder << " " << threadNns[j].numSamples << std::endl;
      remainder--;
      threadNns[j].dstIps = nn.dstIps + 24 * idx;
      threadNns[j].switchIps = nn.switchIps + 24 * idx;
      threadNns[j].queries = nn.queries + 24 * idx;
      idx += threadNns[j].numSamples;
      rc = pthread_create(&threads[j], nullptr, mpForwardAvx, (void *)&args[j]);
    }
    for(auto &t : threads) pthread_join(t, &status);

    nn.numSamples = 112;
    wholes = nn.numSamples / NUM_THREADS;
    remainder = nn.numSamples % NUM_THREADS;
    idx = 0;
    for(int j = 0; j < NUM_THREADS; j++) {
      threadNns[j].numSamples = wholes + (int)(remainder > 0);
      remainder--;
      threadNns[j].dstIps = nn.dstIps + 24 * idx;
      threadNns[j].switchIps = nn.switchIps + 24 * idx;
      threadNns[j].queries = nn.queries + 48 * idx;
      idx += threadNns[j].numSamples;
      rc = pthread_create(&threads[j], nullptr, mpForwardAvx, (void *)&args[j]);
    }
    for(auto &t : threads) pthread_join(t, &status);
  }
  auto stop  = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  double avg = diff.count() / 100;
  std::cout << "Execution took on average " << avg * 1000 << "ms" << std::endl;

  free(predictions);
  free(intBuf);
  free(buf1);
  free(buf2);
  free(nn.hnsas);
  free(nn.allIps);
  free(nn.dstIps);
  free(nn.switchIps);
  free(nn.queries);
  free(nn.nnParams);
}
*/


void measureForward() {
  std::cout << std::endl <<  "Measure Forward" << std::endl << "==============================" << std::endl;
  struct NnConfig config;
  struct NeuralNet nn;
  nn.numSamples = 127;
  nn.numSwitches = 80;
  nn.config = &config;

  auto predictions = (int*)calloc(127     , sizeof(int));
  auto intBuf      = (int*)calloc(127 * 10, sizeof(int));

  auto buf1        = (float*)calloc(127 * 1000, sizeof(float));
  auto buf2        = (float*)calloc(127 * 1000, sizeof(float));
  nn.hnsas         = (float*)calloc(127 * 128 , sizeof(float));
  nn.allIps        = (float*)calloc(127 * 24  , sizeof(float));
  nn.dstIps        = (float*)calloc(127 * 24  , sizeof(float));
  nn.switchIps     = (float*)calloc(127 * 24  , sizeof(float));
  nn.queries       = (float*)calloc(127 * 48  , sizeof(float));
  nn.nnParams      = (float*)calloc(270000    , sizeof(float));

  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 100; i++) {
    nn.numSamples = 123;
    forward(&nn, predictions, buf1, buf2, intBuf);
    nn.numSamples = 112;
    forward(&nn, predictions, buf1, buf2, intBuf);
  }
  auto stop  = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  double avg = diff.count() / 100;
  std::cout << "Execution took on average " << avg * 1000 << "ms" << std::endl;

  free(predictions);
  free(intBuf);
  free(buf1);
  free(buf2);
  free(nn.hnsas);
  free(nn.allIps);
  free(nn.dstIps);
  free(nn.switchIps);
  free(nn.queries);
  free(nn.nnParams);
}


void initCurLocDstAndQuery(struct NeuralNet &nn) {
  float curLoc[] = {0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1.00000};
  float dst[]    = {0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,1.00000,0.00000};

  for(int i = 0; i < nn.numSamples; i++) {
    for(int j = 0; j < 24; j++) {
      nn.switchIps[24 * i + j] = curLoc[j];
      nn.dstIps[24 * i + j]    = curLoc[j];
      nn.queries[48 * i + j]   = curLoc[j];
      nn.queries[48 * i + 24 + j] = dst[j];
    }
  }
}


void testForwardAvx() {
  float logits_z[] = {-2.71743,-1.38464,-1.24700,-1.44149,-1.40630,-1.58803,0.53398,-1.87378,-1.3879};
  int index_z = 6;
  char filename[] = "/home/sim/avxnn/parameters-forwarding-module-stride-16.bin";
  // char filename[] = "/home/sim/git-repos/ldvr/ccode/parameters-forwarding-module-stride-16.bin";
  // char filename[] = "/home/patrick/Documents/GitHub/lkn/avxnn/parameters-forwarding-module-stride-1.bin";
  struct NnConfig config;
  // adaptNnConfigToAvx(&config, 16);
  struct NeuralNet nn;
  // Num parameters with stride 16: 272744
  // Num parameters with stride 1:  264774
  nn.numParams = 272744;
  nn.numSamples = 2;
  nn.numSwitches = 80;
  nn.config = &config;

  auto predictions = (int*)calloc(128     , sizeof(int));
  auto intBuf      = (int*)calloc(128 * 80, sizeof(int));

  auto buf1        = (float*)calloc(128 * 1000, sizeof(float));
  auto buf2        = (float*)calloc(128 * 1000, sizeof(float));
  nn.hnsas         = (float*)calloc(80  * 128 , sizeof(float));
  nn.allIps        = (float*)calloc(80  * 32  , sizeof(float));
  nn.dstIps        = (float*)calloc(128 * 32  , sizeof(float));
  nn.switchIps     = (float*)calloc(128 * 32  , sizeof(float));
  nn.queries       = (float*)calloc(128 * 48  , sizeof(float));
  nn.nnParams      = (float*)calloc(nn.numParams, sizeof(float));
  nn.intHnsas      = (uint64_t*)calloc(80, sizeof(uint64_t));

  fillAllIps(nn.allIps);
  std::cout << "Start filling the HLSA stuff" << std::endl;
  // printRowMajorMatrix(nn.allIps, 80, 24);
  initHnsas(nn.intHnsas);
  std::cout << "Set the Initial Integer values" << std::endl;
  setHnsas(nn.intHnsas, nn.hnsas, 80);
  std::cout << "Set the Initial Integer values" << std::endl;
  // printRowMajorMatrix(nn.hnsas, 80, 128);
  if(readParams(filename, nn.nnParams, nn.numParams) != 0) {
    std::cerr << "Could not read NN parameters!" << std::endl;
    return;
  }
  std::cout << "Successfully read parameters from file" << std::endl;
  forwardAvx(
          &nn,
          predictions,
          buf1,
          buf2,
          intBuf
  );
  std::cout << std::endl << "predictions are" << std::endl;
  //printVector(predictions, 128);
  if(predictions[0] != index_z) {
    std::cerr << "TestForwardAvx failed. Expected index 0 to be " << index_z << ", but got " << predictions[0] << std::endl;
  } else if(predictions[1] != index_z ){
    std::cerr << "TestForwardAvx failed. Expected index 1 to be " << index_z << ", but got " << predictions[1] << std::endl;
  } else {
    std::cout << "Passed test TestForwardAvx" << std::endl;
  }
  std::cout << std::endl;

  free(predictions);
  free(intBuf);
  free(buf1);
  free(buf2);
  free(nn.hnsas);
  free(nn.allIps);
  free(nn.dstIps);
  free(nn.switchIps);
  free(nn.queries);
  free(nn.nnParams);
  free(nn.intHnsas);
}


void write_measurements(struct measurement_tbl &tbl) {
  std::ofstream out(tbl.filename, std::ios::app | std::ios::binary);
  if(out) {
    out.write(reinterpret_cast<const char *>(tbl.timings), (int) sizeof(int) * tbl.idx * tbl.ncol);
    out.close();
    tbl.idx = 0;
  } else {
    std::string s(tbl.filename);
    std::cerr << "Could not open output file " << s << std::endl;
  }
}


void take_measurement(struct measurement_tbl &tbl, int idx, int flag, uint64_t dur, int ns) {
  if(tbl.idx < tbl.max_num_ts) {
    tbl.timings[tbl.idx * tbl.ncol + 0] = idx;
    tbl.timings[tbl.idx * tbl.ncol + 1] = flag;
    tbl.timings[tbl.idx * tbl.ncol + 2] = (int) dur;
    tbl.timings[tbl.idx * tbl.ncol + 3] = ns;
    tbl.idx++;
  }
}


int testMeasurementTbl() {
  int retCode = EXIT_SUCCESS;
  struct measurement_tbl tbl;
  tbl.timings = (int*) calloc(12, sizeof(int));
  tbl.max_num_ts = 3;
  take_measurement(tbl, 1, 5, 10, 1);
  if(tbl.idx != 1) {
    std::cerr << "testMeasurement: Wrong index, expected 1 but got " << tbl.idx << std::endl;
    retCode = EXIT_FAILURE;
  }

  if(retCode == EXIT_SUCCESS) {
    take_measurement(tbl, 2, 6, 1, 2);
    if(tbl.idx != 2) {
      std::cerr << "testMeasurement: Wrong index, expected 2 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
  }

  if(retCode == EXIT_SUCCESS) {
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    if(tbl.idx != 3) {
      std::cerr << "testMeasurement: Wrong index, expected 3 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
  }

  if(retCode == EXIT_SUCCESS) {
    write_measurements(tbl);
    if(tbl.idx != 0) {
      std::cerr << "testMeasurement: Wrong index, expected 2 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
    take_measurement(tbl, 3, 3, 3, 3);
    take_measurement(tbl, 3, 3, 3, 3);
    write_measurements(tbl);
  }

  free(tbl.timings);
  return retCode;
}


int runAvxNn() {
  uint64_t
          ts_start_loop,
          ts_last_write,
          ts_some;
  int retval, oldns;
  char filename[] = "/home/sim/avxnn/parameters-forwarding-module-stride-16.bin";
  // char filename[] = "/home/sim/git-repos/ldvr/ccode/parameters-forwarding-module-stride-16.bin";
  // char filename[] = "/home/patrick/Documents/GitHub/lkn/avxnn/parameters-forwarding-module-stride-1.bin";
  struct measurement_tbl measurements;
  struct NnConfig config;
  // adaptNnConfigToAvx(&config, 16);
  struct NeuralNet nn;
  // Num parameters with stride 16: 272744
  // Num parameters with stride 1:  264774
  nn.numParams = 272744;
  nn.numSamples = 2;
  nn.numSwitches = 80;
  nn.config = &config;

  measurements.timings = (int*)calloc(measurements.max_num_ts * measurements.ncol, sizeof(int));
  auto predictions = (int*)calloc(128     , sizeof(int));
  auto intBuf      = (int*)calloc(128 * 80, sizeof(int));
  auto chosenAggs  = (int*)calloc(256     , sizeof(int));

  auto buf1        = (float*)calloc(128 * 1000, sizeof(float));
  auto buf2        = (float*)calloc(128 * 1000, sizeof(float));
  nn.hnsas         = (float*)calloc(80  * 128 , sizeof(float));
  nn.allIps        = (float*)calloc(80  * 32  , sizeof(float));
  nn.dstIps        = (float*)calloc(128 * 32  , sizeof(float));
  nn.switchIps     = (float*)calloc(128 * 32  , sizeof(float));
  nn.queries       = (float*)calloc(128 * 48  , sizeof(float));
  nn.nnParams      = (float*)calloc(nn.numParams, sizeof(float));
  nn.intHnsas      = (uint64_t*)calloc(80 , sizeof(uint64_t));
  nn.intDsts       = (uint64_t*)calloc(128, sizeof(uint64_t));

  if(construct_file_descriptors(&nn) == EXIT_FAILURE) {
    std::cerr << "Error during initialization of file descriptors, abort." << std::endl;
    return EXIT_FAILURE;
  }
  if(init_nn_labels_eBPF(&nn) == EXIT_FAILURE) {
    std::cerr << "Failed to initialize labels" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Initialize all IPs" << std::endl;
  fillAllIps(nn.allIps);
  // printRowMajorMatrix(nn.allIps, 80, 24);
  std::cout << "Initialize all the HNSA vectors from integers." << std::endl;
  initHnsas(nn.intHnsas);
  setHnsas(nn.intHnsas, nn.hnsas, 80);
  // printRowMajorMatrix(nn.hnsas, 80, 128);
  std::cout << "Read the NN parameters from file." << std::endl;
  if(readParams(filename, nn.nnParams, nn.numParams) != 0) {
    std::cerr << "Could not read NN parameters!" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Successfully read parameters from file" << std::endl;

  int iter = -1;
  ts_last_write = get_nsecs();
  while(1) {
    // sleep(1);
    ++iter;
    if(get_nsecs() - ts_last_write > 60U * 1000000000U) { // Write once every minute.
      if(LOGLEVEL_EBPF > 0) {
        std::cout << "Write " << measurements.idx << " to file after "
                  << (double)(get_nsecs() - ts_last_write) / 1000000. << "ms" << std::endl;
      }
      write_measurements(measurements);
      ts_last_write = get_nsecs();
    }
    ts_start_loop = get_nsecs();

    //------------------------------------------------------------------------
    ts_some = get_nsecs();
    readHnsasFromMap(&nn);
    take_measurement(measurements, iter, (int)FLAG_READ_HNSAS, get_nsecs() - ts_some, nn.numSamples);

    //------------------------------------------------------------------------
    if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
    setHnsas(nn.intHnsas, nn.hnsas, nn.numSwitches);
    // std::cout << "HNSA is " << nn.intHnsas[0] << std::endl;
    if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, (int)FLAG_SET_HNSAS, get_nsecs() - ts_some, nn.numSamples);

    //------------------------------------------------------------------------
    if(LOGLEVEL_EBPF > 0) { std::cout << "Gather the dsetination IPS" << std::endl; }
    if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
    gatherDstIps(&nn);
    if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, (int)FLAG_READ_DST_IPS, get_nsecs() - ts_some, nn.numSamples);

    //------------------------------------------------------------------------
    if(LOGLEVEL_EBPF > 0) {
      std::cout << "  -   Found " << nn.numSamples << " active destinations" << std::endl;
      std::cout << "Handle the directly connected hosts" << std::endl;
    }

    //------------------------------------------------------------------------
    if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
    retval = handle_NN_pt1(&nn);
    if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, (int)FLAG_HANDLE_NN_STAGE_1, get_nsecs() - ts_some, nn.numSamples);
    if(retval == EXIT_FAILURE) {
      std::cerr << "Failed to handle the first input stage." << std::endl;
      break;
    }

    //------------------------------------------------------------------------
    if(nn.numSamples > 0) {
      if(LOGLEVEL_EBPF > 0) { std::cout << "Process " << nn.numSamples << std::endl; }
      // Samples exists for which one or more NN calls must be executed.
      // Create NN inputs. Use the allIps tensor for the current location.
      // Create the desstination IP float vectors and the query vector. Use memcpy for this.

      //------------------------------------------------------------------------
      // Next lines inplement the second NN case
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      if(LOGLEVEL_EBPF > 0) { std::cout << "Prepare Input for Stage 1" << std::endl; }
      prepareInputIpsStageOne(&nn);
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_INPUT_PREP_STAGE_1, get_nsecs() - ts_some, nn.numSamples);

      //------------------------------------------------------------------------
      if(LOGLEVEL_EBPF > 0) { std::cout << "Execute forward pass for " << nn.numSamples << " samples" << std::endl; }
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      forwardAvx(
              &nn,
              predictions,
              buf1,
              buf2,
              intBuf
      );
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_FORWARD_STAGE_1, get_nsecs() - ts_some, nn.numSamples);
      //std::cout << "Prediction of stage 1 is " << predictions[0] << std::endl;
      if(predictions[0] == 5) { predictions[0] = 6; predictions[1] = 6; } else { predictions[0] = 5; predictions[1] = 5; }
      //------------------------------------------------------------------------
      if(LOGLEVEL_EBPF > 0) {std::cout << "Update labels Stage One" << std::endl; }
      oldns = nn.numSamples;
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      retval = updateLabelsNnStageOne(&nn, predictions, chosenAggs);
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_ROUTE_PREP_STAGE_1, get_nsecs() - ts_some, oldns);
      if(EXIT_FAILURE == retval) {
        std::cerr << "Failed to update labels at stage one" << std::endl;
        break;
      }

      //------------------------------------------------------------------------
      // Next lines implement the third NN case
      if(LOGLEVEL_EBPF > 0) { std::cout << "Prepare Input for Stage 2" << std::endl; }
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      prepareInputIpsStageTwo(&nn, chosenAggs);
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_INPUT_PREP_STAGE_2, get_nsecs() - ts_some, nn.numSamples);

      //------------------------------------------------------------------------
      if(LOGLEVEL_EBPF > 0) { std::cout << "Execute forward pass for " << nn.numSamples << " samples" << std::endl; }
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      forwardAvx(
              &nn,
              predictions,
              buf1,
              buf2,
              intBuf
      );
      //std::cout << "Prediction of stage 2 is " << predictions[0] << std::endl;
      if(predictions[0] == 5) { predictions[0] = 6; predictions[1] = 6; } else { predictions[0] = 5; predictions[1] = 5; }
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_FORWARD_STAGE_2, get_nsecs() - ts_some, nn.numSamples);

      //------------------------------------------------------------------------
      if(LOGLEVEL_EBPF > 0) { std::cout << "Update Labels Stage Two" << std::endl; }
      oldns = nn.numSamples;
      if(DO_DETAILED_MEASUREMENTS) ts_some = get_nsecs();
      retval = updateLabelsNnStageTwo(&nn, predictions, chosenAggs);
      if(DO_DETAILED_MEASUREMENTS) take_measurement(measurements, iter, FLAG_ROUTE_PREP_STAGE_2, get_nsecs() - ts_some, oldns);
      if(EXIT_FAILURE == retval) {
        std::cerr << "Failed to update labels at stage two" << std::endl;
        break;
      }
    } else {
      std::cout << "No Samples read in ietration " << iter << " - wait 1s" << std::endl;
      sleep(1);
    }
    take_measurement(measurements, iter, FLAG_SINGLE_LOOP, get_nsecs() - ts_start_loop, nn.numSamples);
    if(LOGLEVEL_EBPF > 0) {
      std::cout << "Loop took " << (double)(get_nsecs() - ts_start_loop) / 1000000.
              << "ms" << std::endl << "==========" << std::endl << std::endl;
    }
  }
  std::cout << "Exit Program" << std::endl;

  free(measurements.timings);
  free(chosenAggs);
  free(predictions);
  free(intBuf);
  free(buf1);
  free(buf2);
  free(nn.hnsas);
  free(nn.allIps);
  free(nn.dstIps);
  free(nn.switchIps);
  free(nn.queries);
  free(nn.nnParams);
  free(nn.intDsts);
  free(nn.intHnsas);

  return EXIT_SUCCESS;
}


int main(void) {
  // measure();
  // testLinear();
  // testRelu();
  // measureRelu();
  // measureArgmax();
  // measureFusedGemm();
  // testOuterProduct();
  // measureOuterProduct();
  // testSlicedGemm();
  // testSpacedOutputGemm();
  // testForwardAvx();
  // measureSpacedOutputGemm();
  // measureGemm();
  // measureForward();
  // measureForwardAvx();
  // struct NnConfig config;
  // int nparams = calcNumParams(&config);
  // std::cout << "NN has " << nparams << " Parameters" << std::endl;
  // adaptNnConfigToAvx(&config, 16);
  // nparams = calcNumParams(&config);
  // std::cout << "AVX NN has " << nparams << " Parameters" << std::endl;
  return runAvxNn();
  // measureMpForwardAvx();
  // return 0;
}
