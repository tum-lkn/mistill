#include <linux/bpf.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <fcntl.h>
#include <bpf/libbpf.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

#define USE_GPU false
#define K 8 // k=8 fat tree system
// update: we want a variable number of labels; therefore, the kernel space program needs to know which labels are unnecessary. Those are marked with a "-1"
#define HiN -1 // high number as a placeholder for unneeded labels
#define LaL -1   // last label in the stack; value is irrelevant
#define MAX_HOPS 6
#define own_IP {10,0,0,3}
#define own_pod 0
#define own_switch 0
#define own_host 3
#define MAP_TIME_SIZE 4000000 // 4 mio
#define TIME_TRACK_FLAG 1 // 0 if time should not be tracked
#define MC_IP 239 // multi cast IP address, where the switches send packets to
// define flags for time tracking
#define FLAG_OUT 0
#define FLAG_CONSTRUCT_FDS 5
#define FLAG_INIT_NN_LABELS 6
#define FLAG_FILL_ALL_IPS 7
#define FLAG_INIT_TENSORS 8
#define FLAG_FILL_HNSA 10
#define FLAG_FILL_HNSA_LOOP 11
#define FLAG_FILL_HNSA_EBPF 12
#define FLAG_FILL_HNSA_TENSOR 13
#define FLAG_ITERATION 15
#define FLAG_TENSOR_TO_GPU 16
#define FLAG_TENSOR_TO_INPUT 17
#define FLAG_GATHER_DESTS 20 // plus number of destinations
#define FLAG_NN_EXE 160 // plus number of destinations
#define FLAG_NN_HANDLING_ALL 300 // plus number of destinations
#define FLAG_NN_HANDLING_1 440 // plus number of destinations
#define FLAG_NN_HANDLING_2 580 // plus number of destinations
#define FLAG_NN_HANDLING_3 720 // plus number of destinations

struct bpf_elf_map {
  __u32 type;
  __u32 size_key;
  __u32 size_value;
  __u32 max_elem;
  __u32 flags;
  __u32 id;
  __u32 pinning;
};
union bpf_attr attr = {
        .pathname = (__u64)(char*)"/sys/fs/bpf/tc/globals/",
        .bpf_fd = 0,
        .file_flags = 0
};
union bpf_attr map_attr = {
        .map_fd = 0,
        .key = 0,
        .value = 0
};

class NeuralNet {
public:
  // torch::Device device("cuda:0");
  torch::Tensor hnsa;
  torch::Tensor switch_IP;
  torch::Tensor dest_IP;
  torch::Tensor all_IPs;
  torch::Tensor output;
  std::vector<int> destinations; // variable size vector
  std::vector<int> nn_switches; // variable size vector
  std::vector<int> outputs; // variable size vector
  int num_dst = 0;
  // int num_sw = 0; num_sw must always be equal to num_dst, therefore it is redundant
  bool print_hnsa = false;
  bool print_tensors = false;
  int labels[MAX_HOPS];
  __u32 fd_dst_ips;
  __u32 fd_nn_lbls;
  __u32 fd_net_stt;
  __u32 fd_t_user;
  torch::jit::script::Module module;

  void construct_file_descriptors();
  void gather_dest_IPs();
  void init_nn_labels_eBPF();
  void init_nn_labels_map(int dst_IP);
  void fill_all_IPs();
  void fill_hnsa();
  int hnsa_in_loop(int a, int b, int c, int i);
  void fill_switch_IP();
  void fill_dest_IP();
  void update_nn_labels(int dst_IP);
  void execute_NN();
  void execute_NN_GPU(); // alternate version
  void handle_NN(int dst_IP);
  void handle_NN_pt1();
  void handle_NN_pt2();
  void handle_NN_pt3();
  int slice_argmax_alt(int dim);
  __u64 get_nsecs();
  void track_time(__u64 t_diff, __u64 flag);
};

void NeuralNet::construct_file_descriptors() {
  __u64 t1, t2;
  t1 = get_nsecs();
  char *path = (char*)"/sys/fs/bpf/tc/globals/dest_IPs";
  attr.pathname = (__u64)path;
  fd_dst_ips = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(fd_dst_ips == -1){
    printf("Map dest_IPs does not exist! Abort operation. \n");
    return;
  }
  path = (char*)"/sys/fs/bpf/tc/globals/nn_labels";
  attr.pathname = (__u64)path;
  fd_nn_lbls = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(fd_nn_lbls == -1){
    printf("Map nn_labels does not exist! Abort operation. \n");
    return;
  }
  path = (char*)"/sys/fs/bpf/tc/globals/network_state";
  attr.pathname = (__u64)path;
  fd_net_stt = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(fd_net_stt == -1){
    printf("Map network_state does not exist! Abort operation. \n");
    return;
  }

  if(TIME_TRACK_FLAG==0){
    // we do not need the fds for those maps
  } else {
    path = (char*)"/sys/fs/bpf/tc/globals/t_user";
    attr.pathname = (__u64)path;
    fd_t_user = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
    if(fd_t_user == -1){
      printf("Map t_user does not exist! Abort operation. \n");
      return;
    }
  }

  torch::Device device("cuda:0");
  path = (char*)"../traced-forwarding-module.pt";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(path); // CPU version
    if(USE_GPU){
      module.to(device);
    }
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  std::cout << "file descriptors set successfully" << std::endl;
  t2 = get_nsecs();
  track_time((t2-t1), FLAG_CONSTRUCT_FDS);
}

void NeuralNet::gather_dest_IPs() {
  __u64 t1, t2;
  t1 = get_nsecs();
  int a, b, c, i = 0;
  __u32 n = 0;
  __u64 val;
  int ret, key = 0;
  __u64 current_time = get_nsecs();
  map_attr.map_fd = fd_dst_ips; // set file descriptor

  for(a = 0; a < K; a++){
    for(b = 0; b < K/2; b++){
      for(c = 2; c < (K/2 + 2); c++){
        int key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
        map_attr.key = (__u64)&key;
        map_attr.value = (__u64)&val;
        ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
        if(ret == -1){
          // printf("Reading process of map dest_IPs failed! No information for IP: %d.%d.%d.%d. \n", 10, a, b, c);
          continue;
        }
        if(val  != 0){
          // printf("For IP: %d.%d.%d.%d, the value is: %lld. \n", 10, a, b, c, val);
        }
        if((current_time-val) < 10000000000) { // time in ns // 10s for testing
          // if there was a connection to this host in the last 10s
          destinations.push_back(key);
          i += 1;
        }
      }
    }
  }
  num_dst = i; // current number of active destinations
  // std::cout << i << "active destinations found successfully" << std::endl;
  t2 = get_nsecs();
  track_time((t2-t1), (FLAG_GATHER_DESTS + num_dst));
}

void NeuralNet::init_nn_labels_eBPF() {
  __u64 t1, t2;
  t1 = get_nsecs();
  // initializes the values in the eBPF Map nn_labels with realistic fall-back routes
  int dst_pod, dst_switch, dst_host, j = 0;
  int key = ( dst_host << 24 ) | ( dst_switch << 16 ) | ( dst_pod << 8 ) | 10;
  for(dst_pod = 0; dst_pod < K; dst_pod++){
    for(dst_switch = 0; dst_switch < K/2; dst_switch++){
      for(dst_host = 2; dst_host < (K/2 + 2); dst_host++){
        int key = ( dst_host << 24 ) | ( dst_switch << 16 ) | ( dst_pod << 8 ) | 10;
        if(own_pod == dst_pod){
          // same pod
          if(own_switch == dst_switch){
            // same ToR
            labels[0] = dst_host - 1; // hosts 2-5 but labels 1-4
            labels[1] = LaL;
            for(j=2; j<6; j++){
              labels[j] = HiN;
            }
            init_nn_labels_map(key); // set MPLS labels for this IP
          } else {
            // same pod other ToR
            labels[0] = own_switch + 5; // the aggregation switch directly above the own host ip+4 and +1 to have a valid label
            labels[1] = dst_switch + 1; // switches 0-3 but labels 1-4
            labels[2] = dst_host - 1; // hosts 2-5 but labels 1-4
            labels[3] = LaL;
            for(j=4; j<6; j++){
              labels[j] = HiN;
            }
            init_nn_labels_map(key); // set MPLS labels for this IP
          }
        } else {
          // other pod
          labels[0] = own_switch + 5; // the aggregation switch directly above the own host ip+4 and +1 to have a valid label
          labels[1] = own_host + 3; // [2-5] + 3 is [5-8] which are the labels, that lead to core switches
          labels[2] = dst_pod + 1; // pods 0-7 but labels 1-8
          labels[3] = dst_switch + 1; // switches 0-3 but labels 1-4
          labels[4] = dst_host - 1; // hosts 2-5 but labels 1-4
          labels[5] = LaL;
          init_nn_labels_map(key); // set MPLS labels for this IP
        }
      }
    }
  }
  std::cout << "eBPF map nn labels successfully initialized" << std::endl;
  t2 = get_nsecs();
  track_time((t2-t1), FLAG_INIT_NN_LABELS);
}

void NeuralNet::fill_all_IPs() {
  __u64 t1, t2;
  t1 = get_nsecs();
  if(USE_GPU){
    torch::Device device("cuda:0");
    all_IPs = torch::ones({1, 80, 24},device);
  } else {
    all_IPs = torch::ones({1, 80, 24});
  }
  int a, b, c = 1, i = 0;
  __u32 n = 0;
  __u32 val;
  // TOP-OF-THE-RACK-SWITCHES
  for(a = 0; a < K; a++){
    for(b = 0; b < K/2; b++){
      int key = ( a << 16 ) | ( b << 8 ) | ( c ); // 10.a.b.c equals 10.pod.switch.1 which is 10.0.0.1 for tor0
      for(int j = 0; j < 24; j++) {
        all_IPs[0][i][j] = (int) ((key >> (23-j))&1); // writing the most significant bit to the first entry of the tensor
      }
      i += 1;
    }
  }
  // AGGREGATION SWITCHES
  for(a = 0; a < K; a++){
    for(b = K/2; b < K; b++){
      int key = ( a << 16 ) | ( b << 8 ) | ( c );
      for(int j = 0; j < 24; j++) {
        all_IPs[0][i][j] = (int) ((key >> (23-j))&1);
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
        all_IPs[0][i][j] = (int) ((key >> (23-j))&1);
      }
      i += 1;
    }
  }
  std::cout << "all_IPs tensor filled successfully" << std::endl;
  t2 = get_nsecs();
  // all_IPs.to(device);
  track_time((t2-t1), FLAG_FILL_ALL_IPS);
}

void NeuralNet::fill_hnsa() {
  __u64 t1, t2, t3;
  t1 = get_nsecs();
  // hnsa = torch::ones({1, 80, 128}); // TEST: maybe faster if it is not initialized every time
  t2 = get_nsecs();
  // track_time((t2-t1), FLAG_FILL_HNSA); // unnecessary; makes the evaluation more complicated

  int a, b, c = 1, i = 0, ret;
  // __u32 n = 0;
  print_hnsa = false; // set to true if the hnsa of tor0 should be printed
  // TOP-OF-THE-RACK-SWITCHES
  for(a = 0; a < K; a++){
    for(b = 0; b < K/2; b++){
      ret = hnsa_in_loop(a,b,c,i);
      if(ret==0){return;}
      i += ret;
      if(print_hnsa==true){print_hnsa=false;} // makes sure only hnsa of tor0 is printed, if any
    }
  }
  // AGGREGATION SWITCHES
  for(a = 0; a < K; a++){
    for(b = K/2; b < K; b++){
      ret = hnsa_in_loop(a,b,c,i);
      if(ret==0){return;}
      i += ret;
    }
  }
  // CORE SWITCHES
  a = K;
  for(b = 1; b <= K/2; b++){
    for(c = 1; c <= K/2; c++){
      ret = hnsa_in_loop(a,b,c,i);
      if(ret==0){return;}
      i += ret;
    }
  }
  // std::cout << "HNSA tensor filled successfully" << std::endl;
  t3 = get_nsecs();
  track_time((t3-t1), FLAG_FILL_HNSA);
}

int NeuralNet::hnsa_in_loop(int a, int b, int c, int i) {
  __u64 t1, t2, t3, t4, val;
  t1 = get_nsecs();
  int key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | MC_IP;
  map_attr.map_fd = fd_net_stt; // needs to be in this function, otherwise it does not work
  map_attr.key = (__u64)&key;
  map_attr.value = (__u64)&val;
  int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    printf("Reading process of map network_state failed! No information for IP: %d.%d.%d.%d. \n", MC_IP, a, b, c);
    return 0;
  }
  t2 = get_nsecs();
  track_time((t2-t1), FLAG_FILL_HNSA_EBPF);
  // printf("[network state] IP: %d.%d.%d.%d information: %lld \n", MC_IP, a, b, c, val);
  for(int j = 0; j < 64; j++) {
    hnsa[0][i][2*j] = (float)((val >> (j)) & 1); // testen, ob das mit float funktioniert (es sollte am Ende float sein)
    hnsa[0][i][2*j+1] = 1 - (float)((val >> (j)) & 1);
  }
  if(print_hnsa==true){
    // std::cout << "for switch id: " << i << "  " << hnsa[0][i] << std::endl;
    std::cout << "HNSA for tor0: " << "  " << hnsa[0][i] << std::endl;
  }
  t3 = get_nsecs();
  track_time((t3-t2), FLAG_FILL_HNSA_TENSOR);
  track_time((t3-t1), FLAG_FILL_HNSA_LOOP);
  return 1;
}

void NeuralNet::fill_switch_IP() {
  // the switch IP is built in this program, so the order of the octets is inverse
  // this helps, since the
  int i, j;
  for(i=0; i<num_dst; i++){
    for(int j = 0; j < 24; j++) {
      switch_IP[i][j] = (int) ((nn_switches[i] >> (23-j))&1); // writing the most significant bit to the first entry of the tensor
    }
  }
  // std::cout << "switch IP int: " << nn_switches[i] << "for i: " << i << std::endl;
}

void NeuralNet::fill_dest_IP() {
  // since the destination IP is extracted from sent packets in the eBPF program, their order of octets is correct
  // so 10.pod.switch.host is the IP over 32 bit
  // however, using bit shift operations each octet has to be treated individually
  int i, j;
  for(i=0; i<num_dst; i++){
    // the first octet, so the 10 is not needed for the NN as an input
    for(j = 0; j < 8; j++) {
      // the first eight entries in the tensor are the second octet, so the bits between 8 and 15
      dest_IP[i][j] = (int) ((destinations[i] >> (15-j))&1); // the MSB is at position 15, so we start at 15 and end at 8
    }
    for(j = 0; j < 8; j++) {
      dest_IP[i][j+8] = (int) ((destinations[i] >> (23-j))&1); //similar to the second octet, now the bits from 23 back to 16
    }
    for(j = 0; j < 8; j++) {
      dest_IP[i][j+16] = (int) ((destinations[i] >> (31-j))&1); // as before, the bits form 31 to 24
    }
  }
  // std::cout << "dest IP int: " << destinations[i] << "for i: " << i << std::endl;
}

void NeuralNet::init_nn_labels_map(int dst_IP) {
  map_attr.map_fd = fd_nn_lbls;

  int a = (dst_IP >> 8) 	& 0xff;
  int b = (dst_IP >> 16) 	& 0xff;
  int c = (dst_IP >> 24) 	& 0xff;
  int key, ret;
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
      printf("Updating process of map nn_labels failed! No labels for IP: %d.%d.%d.%d. \n", 10, a, b, c);
    }
  }
}

void NeuralNet::update_nn_labels(int dst_IP) {
  map_attr.map_fd = fd_nn_lbls;

  int a = (dst_IP >> 8) 	& 0xff;
  int b = (dst_IP >> 16) 	& 0xff;
  int c = (dst_IP >> 24) 	& 0xff;
  int key, ret;
  __u32 n = 0;
  __u32 val[6];
  key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
  map_attr.key = (__u64)&key;
  for(n=0; n<6; n++){
    val[n] = 0;
  }
  map_attr.value = (__u64)&val[0];
  ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    printf("Reading process of map nn_labels failed! No labels for IP: %d.%d.%d.%d. \n", 10, a, b, c);
  }
  // printf("[mpls label] IP: %d.%d.%d.%d labels: %d %d %d %d %d %d \n", 10, a, b, c, val[0], val[1], val[2], val[3], val[4], val[5]);
  for(n=0; n<6; n++){
    val[n] = labels[n];
  }
  ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    printf("Reading process of map nn_labels failed! No labels for IP: %d.%d.%d.%d. \n", 10, a, b, c);
  }
  // printf("[mpls label update] IP: %d.%d.%d.%d labels: %d %d %d %d %d %d \n", 10, a, b, c, val[0], val[1], val[2], val[3], val[4], val[5]);
  // std::cout << "eBPF map nn labels updated successfully" << std::endl;
}

void NeuralNet::execute_NN() {
  __u64 t1, t2, t_gpu;
  // Following line disables the gradient computation. See https://stackoverflow.com/questions/65920683/what-is-the-libtorch-equivalent-to-pytorchs-torch-no-grad
  torch::NoGradGuard no_grad;
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> input;
  if(USE_GPU){ // GPU version needs the tensors on the GPU -> they have to be moved
    torch::Device device("cuda:0");
    t1 = get_nsecs();
    torch::Tensor hnsa_gpu = hnsa.to(device);
    torch::Tensor switch_IP_gpu = switch_IP.to(device);
    torch::Tensor dest_IP_gpu = dest_IP.to(device);
    // all_IPs is already on the GPU since it does not change
    t_gpu = get_nsecs();
    track_time((t_gpu-t1),FLAG_TENSOR_TO_GPU);
    t1 = get_nsecs();
    input.push_back(hnsa_gpu); // hnsa - immer diese Größe
    // these both with a slice up to num_dst, since both tensors are initialized to their maximum lenght, which is usually not used to that extent.
    input.push_back(switch_IP_gpu.slice(0,0,num_dst)); // switch_IP - die geupdated werden
    input.push_back(dest_IP_gpu.slice(0,0,num_dst)); // dest_IP - die angegangen werden sollen: {2, 24} für zwei destinations
    input.push_back(all_IPs); // all_IPs - immer diese Größe
  } else {
    t1 = get_nsecs();
    input.push_back(hnsa); // hnsa - immer diese Größe
    // these both with a slice up to num_dst, since both tensors are initialized to their maximum lenght, which is usually not used to that extent.
    input.push_back(switch_IP.slice(0,0,num_dst)); // switch_IP - die geupdated werden
    input.push_back(dest_IP.slice(0,0,num_dst)); // dest_IP - die angegangen werden sollen: {2, 24} für zwei destinations
    input.push_back(all_IPs); // all_IPs - immer diese Größe
  }
  if(print_tensors==true){ // only for debugging; normally turned off
    std::cout << "HNSA into NN: " << "  " << hnsa[0] << std::endl;
    std::cout << "switch_IP into NN: " << "  " << switch_IP.slice(0,0,num_dst) << std::endl;
    std::cout << "dest_IP into NN: " << "  " << dest_IP.slice(0,0,num_dst) << std::endl;
    std::cout << "all_IPs into NN: " << "  " << all_IPs[0] << std::endl;
  }
  t_gpu = get_nsecs();
  track_time((t_gpu-t1),FLAG_TENSOR_TO_INPUT);
  // Execute the model and turn its output into a tensor.
  output = module.forward(input).toTensor();
  t2 = get_nsecs();
  track_time((t2-t1),FLAG_NN_EXE+num_dst);
}

void NeuralNet::handle_NN_pt1() {
  __u64 t1, t2;
  t1 = get_nsecs();
  if(num_dst == 0){
    t2 = get_nsecs();
    track_time((t2-t1), (FLAG_NN_HANDLING_1 + num_dst));
    return;
  }
  int i, j, n=0;
  std::vector<int> d_temp;
  destinations.swap(d_temp); // swaps content from destinations to d_temp

  for(i=0; i<num_dst; i++){
    int dst_pod     = (d_temp[i] >> 8) 	& 0xff; // pod
    int dst_switch  = (d_temp[i] >> 16) 	& 0xff; // switch
    int dst_host    = (d_temp[i] >> 24) 	& 0xff; // host
    if(dst_pod == own_pod){
      if(dst_switch == own_switch){
        // only one ToR switch is needed
        labels[0] = dst_host - 1; // hosts 2-5 but labels 1-4
        labels[1] = LaL;
        for(j=2; j<6; j++){
          labels[j] = HiN;
        }
        update_nn_labels(d_temp[i]); // set MPLS labels for this IP
      } else {
        destinations.push_back(d_temp[i]); // destinations, that need another NN execution
        n += 1;
      }
    } else {
      destinations.push_back(d_temp[i]); // destinations, that need another NN execution
      n += 1;
    }
  }
  t2 = get_nsecs();
  track_time((t2-t1), (FLAG_NN_HANDLING_1 + num_dst));
  num_dst = n; // n destinations remain
  // std::cout << "handle_NN_pt1 executed successfully" << std::endl;
}

void NeuralNet::handle_NN_pt2() {
  __u64 t1, t2;
  t1 = get_nsecs();
  if(num_dst == 0){
    t2 = get_nsecs();
    track_time((t2-t1), (FLAG_NN_HANDLING_2 + num_dst));
    return;
  }
  int i, j, n=0;
  int dst_pod, dst_switch, dst_host;
  for(i=0; i<num_dst; i++){
    int sw_IP = ( own_pod << 16 ) | ( own_switch << 8 ) | ( 1 ); // pod.switch.1 -- optimized for easier bit shift operations
    nn_switches.push_back(sw_IP);
  }
  fill_switch_IP();
  fill_dest_IP();
  // std::cout << "Execute NN for ToR-switch: " << 10 << "." << own_pod << "." << own_switch << ".1" << std::endl;
  execute_NN();
  std::vector<int> d_temp;
  destinations.swap(d_temp); // swaps content from destinations to d_temp

  for(i=0; i<num_dst; i++){
    dst_pod     = (d_temp[i] >> 8) 	& 0xff; // pod
    dst_switch  = (d_temp[i] >> 16) 	& 0xff; // switch
    dst_host    = (d_temp[i] >> 24) 	& 0xff; // host
    if(dst_pod == own_pod){
      // one ToR and one aggregation switch are needed
      // NN only has to be called for the ToR switch // already done
      labels[0] = std::max(1,slice_argmax_alt(i));
      labels[1] = dst_switch + 1; // switches 0-3 but labels 1-4
      labels[2] = dst_host - 1; // hosts 2-5 but labels 1-4
      labels[3] = LaL;
      for(j=4; j<6; j++){
        labels[j] = HiN;
      }
      update_nn_labels(d_temp[i]); // set MPLS labels for this IP
    } else {
      destinations.push_back(d_temp[i]); // destinations, that need another NN execution
      outputs.push_back(std::max(1,slice_argmax_alt(i))); // needed for the IP of the
      // aggregation switch and the labels of the IPs which depend on two NN executions
      n += 1;
    }
  }
  nn_switches.clear(); // reset for pt3
  t2 = get_nsecs();
  track_time((t2-t1), (FLAG_NN_HANDLING_2 + num_dst));
  num_dst = n; // n destinations remain
  // std::cout << "handle_NN_pt2 executed successfully" << std::endl;
}

void NeuralNet::handle_NN_pt3() {
  __u64 t1, t2;
  t1 = get_nsecs();
  if(num_dst == 0){
    t2 = get_nsecs();
    track_time((t2-t1), (FLAG_NN_HANDLING_3 + num_dst));
    return;
  }
  int i, j, n=0;
  int dst_pod, dst_switch, dst_host, sw_IP;
  for(i=0; i<num_dst; i++){
    // outputs[i] is the label 0-8, but the IPs are 0-7 {label 0 is unused, so 1-8 correspond to 0-7}; therefore the "- 1"
    sw_IP = ( own_pod << 16 ) | ( (outputs[i] - 1) << 8 ) | ( 1 ); // pod.switch.1 -- optimized for easier bit shift operations
    nn_switches.push_back(sw_IP);
    // std::cout << "Execute NN for aggregation switch: " << 10 << "." << own_pod << "." << (outputs[i] - 1) << ".1" << std::endl;
  }
  fill_switch_IP();
  fill_dest_IP();
  execute_NN();
  for(i=0; i<num_dst; i++){
    dst_pod     = (destinations[i] >> 8) 	& 0xff; // pod
    dst_switch  = (destinations[i] >> 16) 	& 0xff; // switch
    dst_host    = (destinations[i] >> 24) 	& 0xff; // host
    // one ToR and one aggregation switch are needed
    // NN only has to be called for the ToR switch // already done
    labels[0] = outputs[i]; // the label found in the previous iteration of the NN
    labels[1] = std::max(1,slice_argmax_alt(i));
    labels[2] = dst_pod + 1; // pods 0-7 but labels 1-8
    labels[3] = dst_switch + 1; // switches 0-3 but labels 1-4
    labels[4] = dst_host - 1; // hosts 2-5 but labels 1-4
    labels[5] = LaL;
    update_nn_labels(destinations[i]); // set MPLS labels for this IP
  }
  // reset vectors for next NN iteration
  destinations.clear();
  outputs.clear();
  nn_switches.clear();
  // std::cout << "handle_NN_pt3 executed successfully" << std::endl;
  t2 = get_nsecs();
  track_time((t2-t1), (FLAG_NN_HANDLING_3 + num_dst));
}

int NeuralNet::slice_argmax_alt(int j) {
  // output.slice() did not work as expected, so this is a workaround
  std::vector<float> o;
  int i, a;
  for(i=0; i<9; i++){
    o.push_back(output[j][i].item<float>());
  }
  // std::cout << "output slice for dim " << j << " is " << o << std::endl;
  a = std::distance(o.begin(), std::max_element(o.begin(),o.end()));
  return a;
}

__u64 NeuralNet::get_nsecs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000UL + ts.tv_nsec;
  // UL stands for unsigned long and makes sure, the 1e9 is read correctly
}


void NeuralNet::track_time(__u64 t_diff, __u64 flag){
  if(TIME_TRACK_FLAG==0){return;}
  map_attr.map_fd = fd_t_user;
  int ret;
  __u64 key = 0, k3, val[2], v[2]; // Key zero contains the current number of entries in the map, necessary to get the next entry to write to.
  map_attr.key = (__u64)&key;
  map_attr.value = (__u64)&val[0]; // Value to key zero --> first element number of elements in the map.
  ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){ // first entry in the map, initialize with value of 1 and the current time.
    map_attr.key = (__u64)&key;
    v[0] = flag;
    v[1] = 1;
    map_attr.value = (__u64)&v;
    ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr)); // index
    k3 = 1;
    map_attr.key = (__u64)&k3;
    v[0] = flag;
    v[1] = t_diff;
    map_attr.value = (__u64)&v;
    ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr)); // time
    // trace_printk("[time] %d", t_diff); // debugging
  } else {
    if(val[1] > MAP_TIME_SIZE){
      // user_time_read("test__");
      std::cout << "MAP t_user has reset!" << std::endl;
      do{
        val[1] = val[1] - MAP_TIME_SIZE; // if the size limit is reached, the first values are overwritten again
      }while(val[1] > MAP_TIME_SIZE); // this actually resets the map also for the program that reads it out afterwards
    }
    k3 = val[1] +1;
    map_attr.key = (__u64)&k3;
    v[0] = flag;
    v[1] = t_diff;
    map_attr.value = (__u64)&v;
    ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr)); // time
    map_attr.key = (__u64)&key;
    v[0] = flag;
    v[1] = k3;
    map_attr.value = (__u64)&v;
    ret = syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, &map_attr, sizeof(map_attr)); // index
    // trace_printk("[time] Duration: %d", t_diff); // debugging
  }
}

int main() {
  std::cout << "Start Program" << std::endl;
  NeuralNet NN;
  __u64 t1, t2, t3;
  NN.construct_file_descriptors();
  NN.init_nn_labels_eBPF();
  NN.fill_all_IPs();
  t1 = NN.get_nsecs();
  NN.dest_IP = torch::ones({128, 24});
  // originally 80x24, but we need to define a switch for each end-host.
  // Therefore this has to be 128 as well. (like dest_IP)
  NN.switch_IP = torch::ones({128, 24});
  NN.hnsa = torch::ones({1, 80, 128});
  t2 = NN.get_nsecs();
  NN.track_time((t2-t1), FLAG_INIT_TENSORS);
  int ret;
  std::cout << "NN running..." << std::endl;
  while(1){
    t1 = NN.get_nsecs();
    NN.fill_hnsa();
    NN.gather_dest_IPs();
    t2 = NN.get_nsecs();
    NN.handle_NN_pt1();
    NN.handle_NN_pt2();
    NN.handle_NN_pt3();
    t3 = NN.get_nsecs();
    NN.track_time((t3-t2),FLAG_NN_HANDLING_ALL);
    NN.track_time((t3-t1),FLAG_ITERATION);

    // return 1; // only one iteration for testing
    // std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 200
  }
  return 1;
}
