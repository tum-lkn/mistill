#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <time.h>
#include <chrono>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <sys/msg.h>
#include <errno.h>
#include <pwd.h>
#include <immintrin.h>

#define DEBUG 0
#define UPDATE_TOR0_ONLY 0
#define MANIPULATE_LINKS 0
const float BW = 100000000;  // Number of Mbit/s on each link.

// Set link up/down : sudo ip link set agg0-eth5 (up | down)


struct eth_stats {
  long mtype;
  char iface_name[14];
  int prev_bytes_out = 0;
  int prev_bytes_in = 0;
  float rate_received = 0;
  float rate_transmitted = 0;
  bool up_receive = true;
  bool up_transmit = true;
  struct timespec * prev_t;
};

uint64_t get_nsecs() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000UL + ts.tv_nsec;
  // UL stands for unsigned long and makes sure, the 1e9 is read correctly
}


/*
float * gemmavx(float * w, float * x, float * y, int bs, int dim_in, int dim_in_avx, int dim_out) {
  // Implements a linear layer that computes xA + b.
  // x is a row major matrix of shape (bs, dim_in).
  // A is a column major matrix of shape (dim_in, dim_out).
  // b is a vector of shape (dim_out,).
  // The output is row major and has a shape of (bs, dim_out).
  // The matrix A and the bias b are stored in the argument w. Argument w
  // contains the matrix A first and then the bias b.
  float * tmp;
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      /* Initialize the sum with a 16th of the bias for the first output
       * neuron. At the end, the 16th will be added together, resulting
       * in the full bias again. In this way, the function saves the
       * addition of the bias at the end and incorporates it into the
       * AVX code.
       *//*
      __m256 sumx16 = _mm256_set1_ps(w[dim_in_avx * dim_out + k] / 8);
      for(int j = 0; j < dim_in; j += 8) {
        // std::cout << "i = " << i << ", k = " << k << " j = " << j << "  -  ";
        // std::cout << "y[" << i * dim_out + k << "] += w[" << k * dim_in + j << "] * x[" << i * dim_in + j << "]" << std::endl;
        __m256 w_part = _mm256_loadu_ps(w + (k * dim_in_avx + j));
        __m256 x_part = _mm256_loadu_ps(x + (i * dim_in + j));

        /* __m512 _mm512_fmadd_ps (__m512 a, __m512 b, __m512 c):
         * Multiply packed single-precision (32-bit) floating-point
         * elements in a and b, add the intermediate result to packed
         * elements in c, and store the results in dst
         *//*
        sumx16 = _mm256_fmadd_ps(w_part, x_part, sumx16);
      }
      tmp = (float*)&sumx16;
      y[i * dim_out + k] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
  }
  return w + dim_in_avx * dim_out + dim_out;
}


float * directGemmAvx(float * w, struct eth_stats * stats, float * y, int bs, int dim_in, int dim_in_avx, int dim_out) {
  int idx;
  float * tmp;
  for(int i = 0; i < bs; i++) {
    for(int k = 0; k < dim_out; k++) {
      __m256 sumx16 = _mm256_set1_ps(w[dim_in_avx * dim_out + k] / 8); // Set bias.
      for(int j = 0; j < 8; j+=4) {
        __m256 w_part = _mm256_loadu_ps(w + (k * dim_in_avx));
        __m256 x_part = _mm256_set_ps(
                (float) (stats[j + 0].up_receive),
                1 - (float) (stats[j + 0].up_receive),
                (float) (stats[j + 0].up_transmit),
                1 - (float) (stats[j + 0].up_transmit),
                stats[j + 0].rate_transmitted,
                stats[j + 0].rate_received,

                (float) (stats[j + 1].up_receive),
                1 - (float) (stats[j + 1].up_receive)
        );
        sumx16 = _mm256_fmadd_ps(w_part, x_part, sumx16);

        w_part = _mm256_loadu_ps(w + (k * dim_in_avx));
        x_part = _mm256_set_ps(
                (float) (stats[j + 1].up_transmit),
                1 - (float) (stats[j + 1].up_transmit),
                stats[j + 1].rate_transmitted,
                stats[j + 1].rate_received,

                (float) (stats[j + 2].up_receive),
                1 - (float) (stats[j + 2].up_receive),
                (float) (stats[j + 2].up_transmit),
                1 - (float) (stats[j + 2].up_transmit)
        );
        sumx16 = _mm256_fmadd_ps(w_part, x_part, sumx16);

        w_part = _mm256_loadu_ps(w + (k * dim_in_avx));
        x_part = _mm256_set_ps(
                stats[j + 2].rate_transmitted,
                stats[j + 2].rate_received,

                (float) (stats[j + 3].up_receive),
                1 - (float) (stats[j + 3].up_receive),
                (float) (stats[j + 3].up_transmit),
                1 - (float) (stats[j + 3].up_transmit),
                stats[j + 3].rate_transmitted,
                stats[j + 3].rate_received
        );
        sumx16 = _mm256_fmadd_ps(w_part, x_part, sumx16);
      }
      tmp = (float*)&sumx16;
      y[i * dim_out + k] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
  }
  return w + dim_in_avx * dim_out + dim_out;
}*/

void relu(float * x, int length) {
  for(int i = 0; i < length; i++) {
    x[i] = ((float)(x[i] > 0)) * x[i];
  }
}


float * directGemm(float * w, const struct eth_stats * stats, float * y, int bs, int dim_in, int dim_out) {
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
      for(int j = 0; j < 8; j++) {
        y[i * dim_out + k] += w[k * dim_in + 6 * j    ] * (float)stats[j].up_receive;
        y[i * dim_out + k] += w[k * dim_in + 6 * j + 1] * (1 - (float) (stats[j].up_receive));
        y[i * dim_out + k] += w[k * dim_in + 6 * j + 2] * (float) (stats[j].up_transmit);
        y[i * dim_out + k] += w[k * dim_in + 6 * j + 3] * (1 - (float) (stats[j].up_transmit));
        y[i * dim_out + k] += w[k * dim_in + 6 * j + 4] * (float)stats[j].rate_transmitted;
        y[i * dim_out + k] += w[k * dim_in + 6 * j + 5] * (float)stats[j].rate_received;
      }
    }
      for(int j = 0; j < dim_out; j++) {
        y[i * dim_out + j] += w[dim_in * dim_out + j];
      }
  }
  return w + dim_in * dim_out + dim_out;
}


void direct_make_hnsa_content(const float * output, uint64_t * hnsa, int num_bins, int bin_size) {
  memset(hnsa, 0, num_bins * sizeof(uint64_t));
  for(int i = 0; i < num_bins; i++) {
    int idx = 0;
    for(int j = bin_size - 1; j >= 0; j--) {
      hnsa[i] <<= 1;
      if(output[i * bin_size + 2 * j] > output[i * bin_size + 2 * j + 1]) {
        hnsa[i] += 1U;
      }
      // hnsa[i] += (uint64_t)output[0][0][i * bin_size + 2 * j].item<float>();
      idx++;
    }
  }
}


float calc_rate(float prev_val, float cur_val, struct timespec * prev_t, struct timespec *cur_t) {
  float duration = (float)(cur_t->tv_sec - prev_t->tv_sec) + (float)(cur_t->tv_nsec - prev_t->tv_nsec) / (float)1e9;
  float cur_rate = (cur_val - prev_val) / duration;
  return cur_rate;
}


float calc_utilization(int prev_val, int cur_val, struct timespec * prev_t, struct timespec *cur_t) {
  float duration = (float)(cur_t->tv_sec - prev_t->tv_sec) + (float)(cur_t->tv_nsec - prev_t->tv_nsec) / (float)1e9;
  int difference = 8 * (cur_val - prev_val);  // Calculates the number of transmitted bits.
  float cur_rate = ((float)difference) / duration / BW; // utilization
  //std::cout <<  "\t(" << cur_val << " - " << prev_val << ") / " << duration << " = " << cur_rate << std::endl;
  return cur_rate;
}


float calc_ewma(float prev_val, float cur_val, float alpha) {
  float ewma = alpha * cur_val + (1 - alpha) * prev_val;
  //std::cout << "\tEWMA: " << alpha << " * " << cur_val << " + (1 - " << alpha << ") * " << prev_val << " = " << ewma << std::endl;
  return ewma;
}


void free_stats(eth_stats * stats, int num_values) {
  for(int i = 0; i < num_values; i++) {
    free(stats[i].prev_t);
  }
  free(stats);
}


int read_tx_bytes(char * interface, int &bytes) {
  char prefix[] = "/sys/class/net/";
  // char prefix[] = "/home/mininet/dc-emulation/cpp/data/";
  char suffix[] = "/statistics/tx_bytes";
  char file_name[80];
  int ret;
  std::ifstream myfile;
  strcpy(file_name, prefix);
  strcpy(file_name + strlen(prefix), interface);
  strcpy(file_name + strlen(file_name), suffix);
  myfile.open(file_name);
  if(myfile.is_open()) {
    char* endptr;
    std::string line;
    getline(myfile, line);
    bytes = (int)strtol(line.c_str(), &endptr, 10);
    myfile.close();
    ret = 0;
  } else {
    bytes = 0;
    std::cerr << "Could not open the file " << file_name << " to read tx bytes." << std::endl;
    ret = 1;
  }
  return ret;
}


int read_rx_bytes(char * interface, int &bytes) {
  char prefix[] = "/sys/class/net/";
  // char prefix[] = "/home/mininet/dc-emulation/cpp/data/";
  char suffix[] = "/statistics/rx_bytes";
  char file_name[80];
  int ret;
  std::ifstream myfile;
  strcpy(file_name, prefix);
  strcpy(file_name + strlen(prefix), interface);
  strcpy(file_name + strlen(file_name), suffix);
  myfile.open(file_name);
  if(myfile.is_open()) {
    char* endptr;
    std::string line;
    getline(myfile, line);
    bytes = (int)strtol(line.c_str(), &endptr, 10);
    myfile.close();
    ret = 0;
  } else {
    bytes = 0;
    std::cerr << "Could not open the file " << file_name << " to read rx bytes." << std::endl;
    ret = 1;
  }
  return ret;
}


int read_availability(char * interface, bool &available) {
  char prefix[] = "/sys/class/net/";
  // char prefix[] = "/home/mininet/dc-emulation/cpp/data/";
  char suffix[] = "/operstate";
  char file_name[80];
  char symbol;
  int ret;
  std::ifstream myfile;
  strcpy(file_name, prefix);
  strcpy(file_name + strlen(prefix), interface);
  strcpy(file_name + strlen(file_name), suffix);
  myfile.open(file_name);
  if(myfile.is_open()) {
    myfile.get(symbol);
    if(symbol == 'u') {
      available = true;
    } else {
      available = false;
    }
    ret = 0;
    myfile.close();
  } else {
    available = false;
    std::cerr << "Could not open the file " << file_name << " to read availability." << std::endl;
    ret = 1;
  }
  return ret;
}


void init_node(eth_stats * stats, int k, char * node) {
  for(int j = 1; j < k + 1; j++) {
    int idx = j - 1;
    stats[idx].rate_received = 0;
    stats[idx].rate_transmitted = 0;
    stats[idx].up_receive = true;
    stats[idx].up_transmit = true;
    stats[idx].prev_t = (struct timespec*)malloc(sizeof(struct timespec));
    clock_gettime(CLOCK_REALTIME, stats[idx].prev_t);
    strcpy(stats[idx].iface_name, node);
    strcpy(stats[idx].iface_name + strlen(stats[idx].iface_name), "-eth");
    strcpy(stats[idx].iface_name + strlen(stats[idx].iface_name), std::to_string(j).c_str());
    read_rx_bytes(stats[idx].iface_name, stats[idx].prev_bytes_in);
    read_tx_bytes(stats[idx].iface_name, stats[idx].prev_bytes_out);
  }
}


void init_stats(eth_stats * stats, int k, int max_num_pods) {
  int idx = 0;
  int node_idx = 2;
  int num_pods;
  if(k < max_num_pods) { num_pods = k; } else { num_pods = max_num_pods; }

  for(int i = 0; i < num_pods * k / 2; i++) {
    char name[12];
    strcpy(name, "tor");
    strcpy(name + strlen(name), std::to_string(i).c_str());
    init_node(stats + idx, k, name);
    stats[idx].mtype = node_idx;
    idx += k;
    node_idx++;
  }
  for(int i = 0; i < num_pods * k / 2; i++) {
    char name[12];
    strcpy(name, "agg");
    strcpy(name + strlen(name), std::to_string(i).c_str());
    init_node(stats + idx, k, name);
    stats[idx].mtype = node_idx;
    idx += k;
    node_idx++;
  }
  for(int i = 0; i < k * k / 4; i++) {
    char name[12];
    strcpy(name, "core");
    strcpy(name + strlen(name), std::to_string(i).c_str());
    init_node(stats + idx, k, name);
    stats[idx].mtype = node_idx;
    idx += k;
    node_idx++;
  }
}


/*
 * Attributes in the /proc/net/dev file are:
 * 1. Interface
 * Incoming direction (receiving)
 * 2. bytes
 * 3. packets
 * 4. errs
 * 5. dropped
 * 6. fifo
 * 7. frame
 * 8. compressed
 * 9. multicast
 * Outgoing direction (transmitting)
 * 10. bytes
 * 11. packets
 * 12. errs
 * 13 drop
 * 14. fifo
 * 15. colls
 * 16. carrier
 * 17. compressed
 */
void from_proc_net_dev(int k, eth_stats * stats) {
    std::string line;
    std::ifstream myfile ("/proc/net/dev");
    // std::ifstream myfile ("sample.txt");
    if (myfile.is_open()) {
        char symbol;
        char buffer[100];
        int line_num = 0;
        int attribute_num = 0;
        int parsing_token = 0;
        int idx = 0;
        int token_idx = 0;
        int fast_forward = 0;
        int tmp_int;
        float tmp_float;
        struct timespec tv{};
        clock_gettime(CLOCK_REALTIME, &tv);
        // int num_structs = (int)(5 * k * k * (k + 1) / 4);
        // eth_stats stats[num_structs];

        while(myfile.get(symbol)) {
            if(symbol == '\n') {
                if (!fast_forward && line_num >= 2 && parsing_token) {
                    // Still parsing a token. Stop parsing and store the token.
                    attribute_num++;
                    buffer[token_idx] = '\0';
                    parsing_token = 0;
                    std::cout << buffer << " - ";
                    std::cout << '\n';
                }
                line_num++;
                attribute_num = 0;
                fast_forward = 0;
                continue;
            }

            if(fast_forward) continue;

            if(line_num < 2 || symbol == ':') continue;

            if(symbol == ' ' && parsing_token) {
                // New whitespace after a token started.
                parsing_token = 0;
                attribute_num++;
                buffer[token_idx] = '\0';
                if((token_idx < 9 || (buffer[0] == 'o' && buffer[1] == 'v' &&
                                      buffer[2] == 's')) && attribute_num == 1) {
                    // The first attribute is the interface name. Make sure to get the
                    // switch interfaces and ignore all others. If its an undesired
                    // interface name skip the rest of the line.
                    fast_forward = 1;
                } else {
                    std::cout << buffer << " - ";
                    if(attribute_num == 1) {
                        // Copy the interface name.
                        strcpy(stats[idx].iface_name, buffer);
                    }
                    if(attribute_num == 2) {
                        // Get the cumulative sum of the received bytes.
                        char* endptr;
                        tmp_int = (int)strtol(buffer, &endptr, 10);
                        tmp_float = calc_utilization((float)stats[idx].prev_bytes_in, (float)tmp_int, stats[idx].prev_t, &tv);
                        stats[idx].prev_bytes_in = tmp_int;
                        stats[idx].rate_received = calc_ewma(stats[idx].rate_received, tmp_float, 0.9);
                    }
                    if(attribute_num == 10) {
                        // Get the cumulative sum of the send bytes. Last feature to get,
                        // thus skip the rest of the line.
                        char* endptr;
                        tmp_int = (int)strtol(buffer, &endptr, 10);
                        tmp_float = calc_utilization((float)stats[idx].prev_bytes_out, (float)tmp_int, stats[idx].prev_t, &tv);
                        stats[idx].prev_bytes_out = tmp_int;
                        stats[idx].rate_transmitted = calc_ewma(stats[idx].rate_transmitted, tmp_float, 0.9);
                        // Update the time stamp here, its the last value that we read.
                        stats[idx].prev_t->tv_nsec = tv.tv_nsec;
                        stats[idx].prev_t->tv_sec = tv.tv_sec;

                        fast_forward = 1;
                        std::cout << std::endl;
                        idx++;
                    }
                }
                continue;
            }

            // Ignore whitespaces.
            if(symbol == ' ') continue;

            if(!parsing_token) {
                // First non-whitespace symbol, beginning of a new token
                parsing_token = 1;
                token_idx = 0;
            }
            buffer[token_idx] = symbol;
            token_idx++;
        }
        myfile.close();


    } else {
        std::cout << "Unable to open file";
    }
}


void add_interface_availability(int k, eth_stats * stats) {
    int num_structs = (int)(5 * k * k * (k + 1) / 4);
    char prefix[] = "/sys/class/net/";
    char filename[] = "/operstate";
    char path[39];
    char symbol;
    std::ifstream myfile;
    // /sys/class/net/agg0-eth5/operstate
    for(int i = 0; i < num_structs; i++) {
        strcpy(path, prefix);
        strcpy(path + 15, stats[i].iface_name);
        strcpy(path + 15 + strlen(stats[i].iface_name), filename);
        std::cout << "Get state for interface " << stats[i].iface_name << " in " << path;
        myfile.open(path);
        if(myfile.is_open()) {
            myfile.get(symbol);
            if(symbol == 'u') {
                std::cout << " is up";
                stats[i].up_receive = true;
                stats[i].up_transmit = true;
            } else {
                std::cout << " is down";
                stats[i].up_receive = false;
                stats[i].up_transmit = false;
            }
            myfile.close();
        } else {
            std::cout << " Does not exist! ";
        }
        std::cout << std::endl;
    }
}


void print_stat(eth_stats * stat) {
  std::cout << "Interface " << stat->iface_name << " has received " << stat->prev_bytes_in
            << " Bytes, transmitted " << stat->prev_bytes_out << " Bytes, Sending rate is " << stat->rate_transmitted
            << " Bytes/s, Incoming rate is " << stat->rate_transmitted << " Bytes/s and is available: "  << stat->up_receive << std::endl << "";
}


void set_fix_stats(struct eth_stats * stat, float factor) {
  stat->rate_transmitted = factor;
  stat->rate_received = factor;
  stat->prev_bytes_out = factor * 0;
  stat->prev_bytes_in = factor * 0;
}


void update_stats(struct eth_stats * stat) {
  struct timespec tv{};
  int tx_bytes, rx_bytes;
  bool available;
  float tmp_float, factor;
  clock_gettime(CLOCK_REALTIME, &tv);
  if(DEBUG>0){std::cout << "\tUpdate interface" << stat->iface_name << std::endl;}

  read_availability(stat->iface_name, available);
  stat->up_receive = available;
  stat->up_transmit = available;

  if(MANIPULATE_LINKS == 1){ // only to test the link up down test case
    if(available == false){
      set_fix_stats(stat, -1);
    } else if(strcmp(stat->iface_name, "tor0-eth1") == 0){
      set_fix_stats(stat, 0.05);
    } else if(strcmp(stat->iface_name, "tor0-eth2") == 0){
      set_fix_stats(stat, 0.20);
    } else if(strcmp(stat->iface_name, "tor0-eth3") == 0){
      set_fix_stats(stat, 0.15);
    } else if(strcmp(stat->iface_name, "tor0-eth4") == 0){
      set_fix_stats(stat, 0.10);
    } else if(strcmp(stat->iface_name, "tor0-eth5") == 0){
      set_fix_stats(stat, 0.001);
    } else if(strcmp(stat->iface_name, "tor0-eth6") == 0){
      set_fix_stats(stat, 0.50);
    } else if(strcmp(stat->iface_name, "tor0-eth7") == 0){
      set_fix_stats(stat, 0.85);
    } else if(strcmp(stat->iface_name, "tor0-eth8") == 0){
      set_fix_stats(stat, 0.90);
    } else if(strcmp(stat->iface_name, "agg0-eth1") == 0){
      set_fix_stats(stat, 0.001);
    } else if(strcmp(stat->iface_name, "agg1-eth1") == 0){
      set_fix_stats(stat, 0.50);
    } else if(strcmp(stat->iface_name, "agg2-eth1") == 0){
      set_fix_stats(stat, 0.85);
    } else if(strcmp(stat->iface_name, "agg3-eth1") == 0){
      set_fix_stats(stat, 0.90);
    } else { 
      read_tx_bytes(stat->iface_name, tx_bytes);
      //std::cout << std::endl << "Calculation of rate " << stat->iface_name << ": " << std::endl;
      tmp_float = calc_utilization(stat->prev_bytes_out, tx_bytes, stat->prev_t, &tv);
      stat->prev_bytes_out = tx_bytes;
      stat->rate_transmitted = calc_ewma(stat->rate_transmitted, tmp_float, 0.9);

      read_rx_bytes(stat->iface_name, rx_bytes);
      tmp_float = calc_utilization(stat->prev_bytes_in, rx_bytes, stat->prev_t, &tv);
      stat->prev_bytes_in = rx_bytes;
      stat->rate_received = calc_ewma(stat->rate_received, tmp_float, 0.9);
    }
  } else {
    read_tx_bytes(stat->iface_name, tx_bytes);
    //std::cout << std::endl << "Calculation of rate " << stat->iface_name << ": " << std::endl;
    tmp_float = calc_utilization(stat->prev_bytes_out, tx_bytes, stat->prev_t, &tv);
    stat->prev_bytes_out = tx_bytes;
    stat->rate_transmitted = calc_ewma(stat->rate_transmitted, tmp_float, 0.9);

    read_rx_bytes(stat->iface_name, rx_bytes);
    tmp_float = calc_utilization(stat->prev_bytes_in, rx_bytes, stat->prev_t, &tv);
    stat->prev_bytes_in = rx_bytes;
    stat->rate_received = calc_ewma(stat->rate_received, tmp_float, 0.9);
  }

  if(DEBUG>-1){
    std::cout << "\trx_bytes: " << stat->prev_bytes_out << " tx_bytes: " << stat->prev_bytes_in << " ";
    std::cout << "rate received: " << stat->rate_received << " rate transmitted: " << stat->rate_transmitted << " ";
    std::cout << "time [ns]: " << (tv.tv_sec * 1000000000UL + tv.tv_nsec) << std::endl;
  }
  stat->prev_t->tv_sec = tv.tv_sec;
  stat->prev_t->tv_nsec = tv.tv_nsec;
}


void update_node_stats(struct eth_stats * stats, int k) {
  for (int i = 0; i < k; i++) {
    update_stats(stats + i);
  }
}


void monitor_all_interfaces() {
  struct timespec start{};
  struct timespec end{};
  struct timespec tv{};
  clock_gettime(CLOCK_REALTIME, &start);
  int k = 4;
  int num_structs = (int)(k * k * k * (k + 1) / 4);
  auto stats = (eth_stats*)malloc(sizeof(eth_stats) * num_structs);
  init_stats(stats, k, k);
  do {
    for (int i = 0; i < num_structs; i++) {
      update_stats(stats + i);
      print_stat(stats + i);
      clock_gettime(CLOCK_REALTIME, &tv);
    }
  } while(tv.tv_sec - start.tv_sec < 10);
  free_stats(stats, num_structs);
  clock_gettime(CLOCK_REALTIME, &end);
  std::cout << "Execution took " << (double)(end.tv_sec - start.tv_sec) +(double)(end.tv_nsec - start.tv_nsec) / 1e9 << "s" << std::endl;
}


void monitor_one_interface(char * interface) {
  auto stat = (struct eth_stats*) malloc(sizeof(struct eth_stats));
  stat->prev_t = (struct timespec*)malloc(sizeof(struct timespec));
  strcpy(stat->iface_name, interface);
  struct timespec tv{};
  struct timespec start{};
  clock_gettime(CLOCK_REALTIME, &start);
  do {
      update_stats(stat);
      print_stat(stat);
      clock_gettime(CLOCK_REALTIME, &tv);
      sleep(1);
  } while(tv.tv_sec - start.tv_sec < 60);
  free(stat->prev_t);
  free(stat);
}


void monitor_one_node(char * node, int k) {
  struct timespec start{};
  struct timespec end{};
  struct timespec tv{};
  clock_gettime(CLOCK_REALTIME, &start);
  auto stats = (eth_stats*)malloc(sizeof(eth_stats) * k);
  init_node(stats, k, node);
  do {
    for (int i = 0; i < k; i++) {
      update_stats(stats + i);
      print_stat(stats + i);
      clock_gettime(CLOCK_REALTIME, &tv);
    }
    std::cout << std::endl;
    sleep(1);
  } while(tv.tv_sec - start.tv_sec < 10);
  free_stats(stats, k);
  clock_gettime(CLOCK_REALTIME, &end);
  std::cout << "Execution took " << (double)(end.tv_sec - start.tv_sec) +(double)(end.tv_nsec - start.tv_nsec) / 1e9 << "s" << std::endl;
}


void make_hnsa_content(const torch::Tensor &output, uint64_t * hnsa, int num_bins, int bin_size) {
  memset(hnsa, 0, num_bins * sizeof(uint64_t));
  for(int i = 0; i < num_bins; i++) {
    int idx = 0;
    for(int j = bin_size - 1; j >= 0; j--) {
      hnsa[i] <<= 1;
      hnsa[i] += (uint64_t)output[0][0][i * bin_size + 2 * j].item<float>();
      idx++;
    }
  }
}


void extract_hnsa_bits(float * content, uint64_t * hnsa, int num_bins, int bin_size) {
  for(int i = 0; i < num_bins; i ++) {
    for(int j = 0; j < bin_size; j++) {
      content[i * bin_size + 2 * j]     =     (float)((hnsa[i] >> j) & 1);
      content[i * bin_size + 2 * j + 1] = 1 - (float)((hnsa[i] >> j) & 1);
    }
  }
}


int send_updates(char * node, int k) {
  struct timespec start{};
  struct timespec tv{};
  struct sockaddr_in address{};
  int num_bins = 1;
  int bin_size = 64;

  // Load the Neural Network.
  // std::string model_path = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  std::string model_path = "/home/sim/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  torch::jit::script::Module module;
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  auto hnsa = (uint64_t*)malloc(num_bins * sizeof(u_int64_t));
  auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  torch::Tensor network_states = torch::empty({1, 1, 8, 6}, options);
  torch::Tensor network_masks = torch::ones({1, 1, 8, 1}, options);
  at::Tensor output;
  inputs.emplace_back(network_states);
  inputs.emplace_back(network_masks);
  try {
    module = torch::jit::load(model_path);
    std::cout << "Loaded model" << std::endl;

    // Initialize the interface stat structs.
    auto stats = (eth_stats *) malloc(sizeof(eth_stats) * k);
    init_node(stats, k, node);

    // Initialize the networking stuff.
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    char group[] = "239.0.0.1"; // argv[1];
    int port = 5007;  // atoi(argv[2]);
    int ret;
    if (sock < 0) {
      std::cerr << "Error opening socket" << std::endl;
      exit(EXIT_FAILURE);
    }
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(group);
    address.sin_port = htons(port);
    std::cout << "Configured interface to send multicast messages" << std::endl;

    // Start the loop that fetches new interface statistics, executes the NN and
    // sends the update message.
    clock_gettime(CLOCK_REALTIME, &start);
    do {
      clock_gettime(CLOCK_REALTIME, &tv);
      // Update the interface statistics.
      update_node_stats(stats, k);
      for (int i = 0; i < k; i++) {
        network_states[0][0][i][0] = (float)(stats[i].up_receive);
        network_states[0][0][i][1] = (float)(!stats[i].up_receive);
        network_states[0][0][i][2] = (float)(stats[i].up_transmit);
        network_states[0][0][i][3] = (float)(!stats[i].up_transmit);
        network_states[0][0][i][4] = stats[i].prev_bytes_in;
        network_states[0][0][i][5] = stats[i].prev_bytes_out;
        if(DEBUG>0){print_stat(stats + i);}
      }
      output = module.forward(inputs).toTensor();
      if(DEBUG>0){
        for(int i = 0; i < 128; i++) {
          if(i % 2 == 0) std::cout << " ";
          std::cout << output[0][0][i].item<float>();
        }
        std::cout << std::endl;
        make_hnsa_content(output, hnsa, num_bins, bin_size);
        for(int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1) << (1 - ((hnsa[0] >> i) & 1));
        std::cout << std::endl;
        for(int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1);
        std::cout << std::endl;
      
        // uint32_t tmp;
        // float elapsed = (float) (tv.tv_sec - start.tv_sec);
        // char message[] = "message from 239.0.0.1";
        // memcpy(&tmp, &elapsed, sizeof(tmp));
        // tmp = htonl(tmp);
        // ret = sendto(sock, message, strlen(message), 0, (struct sockaddr *)&address, sizeof(address));
        std::cout << "Send the integer " << hnsa[0] << std::endl;
      }
      ret = sendto(sock, hnsa, sizeof(uint64_t) * num_bins, 0, (struct sockaddr *) &address, sizeof(address));

      if (ret < 0) {
        std::cerr << "Error sending message" << std::endl;
        exit(EXIT_FAILURE);
      }
      sleep(1);
    } while (tv.tv_sec - start.tv_sec < 10);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  free(hnsa);
  return EXIT_SUCCESS;
}


int test_nn() {
  std::string model_path = "/home/sim/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  // std::string model_path = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // module = torch::jit::load(argv[1]);
    module = torch::jit::load(model_path);
    //torch::config::show();
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::abs(torch::randn({1, 1, 8, 6})));
    inputs.push_back(torch::abs(torch::randn({1, 1, 8, 1})));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    if(DEBUG>0){
      std::cout << "Inputs: " << std::endl << inputs << std::endl;
      std::cout << "Outputs: " << std::endl << output << std::endl;
      std::cout << std::endl << "NN output\t" << std::endl;
      for(int i = 0; i < 128; i++) {
        std::cout << output[0][0][i].item<float>() << " ";
      }
      std::cout << std::endl;
    }
    auto hnsa = (uint64_t*)malloc(1 * sizeof(uint64_t));
    auto message = (float*)malloc(2 * 64 * sizeof(float));
    make_hnsa_content(output, hnsa, 1, 64);
    extract_hnsa_bits(message, hnsa, 1, 64);
    if(DEBUG>0){
      for(int i = 0; i < 128; i++) {
        std::cout << message[i] << " ";
      }
      std::cout << std::endl;
    }
    free(hnsa);
    free(message);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  return EXIT_SUCCESS;
}


int initialize_data() {
  std::string model_path = "/home/sim/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  // std::string model_path = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  torch::jit::script::Module module;
  int num_bins = 1;
  int bin_size = 64;
  auto hnsa = (uint64_t*)malloc(num_bins * sizeof(u_int64_t));
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // module = torch::jit::load(argv[1]);
    module = torch::jit::load(model_path);

    // Create a vector of inputs.
    at::Tensor link_state = torch::abs(torch::randn({1, 1, 8, 6}));
    for(int i = 0; i < 8; i++) {
      link_state[0][0][i][0] = 1;
      link_state[0][0][i][1] = 0;
      link_state[0][0][i][2] = 1;
      link_state[0][0][i][3] = 0;
      link_state[0][0][i][4] = link_state[0][0][i][4] / 1000;
      link_state[0][0][i][5] = link_state[0][0][i][5] / 1000;
    }
    std::cout << "LInk state is " << link_state << std::endl;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(link_state);
    inputs.push_back(torch::abs(torch::ones({1, 1, 8, 1})));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << "The outut is: ";
    for (int i = 0; i < 128; i++) {
      // if (i % 2 == 0) std::cout << " ";
      std::cout << output[0][0][i].item<float>();
      if ( i < 127) std::cout << ", ";
    }
    std::cout << std::endl;
    make_hnsa_content(output, hnsa, num_bins, bin_size);
    std::cout << "The HNSA is: ";
    // for (int i = 0; i < 64; i++) std::cout << ", " << ((hnsa[0] >> i) & 1) << (1 - ((hnsa[0] >> i) & 1));
    // std::cout << std::endl;
    for (int i = 0; i < 64; i++) {
      std::cout << ((hnsa[0] >> i) & 1);
      if ( i < 63) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
    std::cout << "The HSNA int representation is: " << hnsa[0] << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  free(hnsa);
  std::cout << "ok\n";
  return EXIT_SUCCESS;
}


void test_vector_assignment() {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  torch::Tensor network_states = torch::empty({1, 1, 8, 6}, options);
  std::cout << network_states << std::endl;
  std::cout << std::endl << "After assignment" << std::endl;
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 6; j ++) {
      // network_states.index_put_({0, i, j}, (float)(i * 8 + j));
      network_states[0][0][i][j] = (float)(i * 8 + j);
    }
  }
  std::cout << network_states << std::endl;
}


void copy_files(int k) {
  // TODO: Implement the sharing of data. Use named pipes for this.
  char src_prefix[] = "/sys/devices/virtual/net/";
  char dst_prefix[] = "/home/mininet/dc-emulation/cpp/data/";
  char name[7];
  for(int pod = 0; pod < k; pod++) {
    for(int sw = 0; sw < k / 2; sw++){
      for(int iface = 1; iface <= k; iface++) {
        // Copy the interface files
        // <path>/operstate
        // <path>/statistics
      }
    }
  }
  for(int cid = 0; cid < k * k / 4; k++) {
    for(int iface = 1; iface <= k; iface++) {
      // Copy the interface files
      // <path>/operstate
      // <path>/statistics
    }
  }
}


void make_queue_path(char * queue_path, char * work_dir) {
  //struct passwd *pw = getpwuid(getuid());
  char queue_name[] = "/msg-queue";
  strcpy(queue_path, work_dir);
  strcpy(queue_path + strlen(queue_path), queue_name);
}


void make_signal_path(char * path, char * work_dir) {
  //struct passwd *pw = getpwuid(getuid());
  char queue_name[] = "/producer-quit";
  strcpy(path, work_dir);
  strcpy(path + strlen(path), queue_name);
}


int producer(int k, int max_num_pods, char * work_dir, int duration) {
  struct timespec t_start{}, t_now{}, t_loop{};
  char queue_path[256];
  make_queue_path(queue_path, work_dir);
  char signal_path[256];
  make_signal_path(signal_path, work_dir);

  std::cout << "Send messages to " << queue_path << std::endl;
  std::cout << "Listen for quti signal on " << signal_path << std::endl;
  // Initialize the interface stat structs.
  std::cout << "Reserve memory" << std::endl;
  int num_pods;
  if(k < max_num_pods) { num_pods = k; } else { num_pods = max_num_pods; }
  int num_switches = k * k / 4 + k * num_pods;
  auto stats = (eth_stats *) malloc(sizeof(eth_stats) * k * num_switches);
  std::cout << "Initialize stats for " << num_switches << " switches" << std::endl;
  init_stats(stats, k, max_num_pods);
  if(DEBUG>0){ for(int i = 0; i < num_switches; i++) { for(int j = 0; j < k; j++) {print_stat(stats + i * k + j); }}}

  key_t key_msg_q = ftok(queue_path, 'b');
  int msg_q_id = msgget(key_msg_q, 0666 | IPC_CREAT);
  if(msg_q_id < 0) {
    std::cerr << "Could not open message queue, error number " << msg_q_id << std::endl;
    free(stats);
    return -1;
  }
  struct msqid_ds q_settings{};
  if(msgctl(msg_q_id, IPC_STAT, &q_settings) < 0) {
    std::cerr << "Failed to retrieve settings of queue " << msg_q_id << ", error is " << errno << std::endl;
  }
  std::cout << "Set Q size from " << q_settings.msg_qbytes << " Bytes to " << 1048576 << " Bytes" << std::endl;
  // q_settings.msg_qbytes = 2 * num_switches * k * sizeof(eth_stats);
  q_settings.msg_qbytes = 1048576 * 100;
  if(msgctl(msg_q_id, IPC_SET, &q_settings) < 0) {
    std::cerr << "Failed to update settings of queue " << msg_q_id << ", error is " << errno;
    std::cerr << " Check /proc/sys/kernel/msgmnb for the maximum limit. If your value exceeds this limit, ";
    std::cerr << "ensure that you have the CAP_IPC_RESOURCE privilege. Check man msgctl for more information.";
  }
  std::cout << "Opened queue with ID " << key_msg_q << std::endl;

  clock_gettime(CLOCK_REALTIME, &t_start);
  clock_gettime(CLOCK_REALTIME, &t_now);
  while(t_now.tv_sec - t_start.tv_sec < duration) {
    clock_gettime(CLOCK_REALTIME, &t_now);
    for(int node_idx = 0; node_idx < num_switches; node_idx++) {
      // If only Tor0 should send updates and the interface does not start with tor0, then
      // ignore this stat struct and continue with the next one.
      if(UPDATE_TOR0_ONLY && (strncmp(stats[node_idx * k].iface_name, "tor0", 4) != 0)) {
        continue;
      } else {
        // Check if the current interface belongs to a core switch. If so, use the update node stats function
        // with the maximum number of pods to avoid reading from interfaces that do not exist.
        if (stats[node_idx * k].iface_name[0] == 'c' || node_idx >= num_pods * k) {
          update_node_stats(stats + node_idx * k, max_num_pods);
        } else {
          update_node_stats(stats + node_idx * k, k);
        }
        if (DEBUG > 0){
          std::cout << "send message with msg id " << (stats + node_idx * k)->mtype << " for node " << node_idx
                    << std::endl;
        }
        if (msgsnd(msg_q_id, (stats + node_idx * k), sizeof(eth_stats) * k, IPC_NOWAIT) < 0) {
          if (errno == EAGAIN) {
            std::cerr << "Error sending message. Queue is full." << std::endl;
          } else {
            std::cerr << "Unexpected error sending message. Error code is " << errno << ". Quit." << std::endl;
            break;
          }
        }
      }
    }
    do { // instead of sleep a busy wait
      clock_gettime(CLOCK_REALTIME, &t_loop);
    } while(((t_loop.tv_sec - t_now.tv_sec) * 1000000000UL + (t_loop.tv_nsec - t_now.tv_nsec)) < 150000000); // default: 10ms/ 150ms
    // UL stands for unsigned long and makes sure, the 1e9 is read correctly
    // waiting times are: 10ms for link up down test case (only tor0 is active)
    // 150ms for regular test, where all switches are active
    // -> otherwise this leads to congestions

    std::ifstream signal(signal_path);
    if(signal.is_open()) {
      std::cout << "Signal received to shut down" << std::endl;
      signal.close();
      remove(signal_path);
      break;
    }
  }
  if(t_now.tv_sec - t_start.tv_sec >= duration) {
    std::cout << "Timeout, stop producer after the configured " << duration << " seconds" << std::endl;
  }
  std::cout << "Quit producer" << std::endl;
  free(stats);
  msgctl(msg_q_id, IPC_RMID, NULL);
  return 0;
}


int consumer(int k) {
  struct timespec t_start{}, t_now{};
  char node[8];
  char queue_path[] = "/home/mininet/dc-emulation/msg-queue";
  int abs_node_num = 1;
  int i = 0;

  // Initialize the interface stat structs.
  auto stats = (eth_stats *) malloc(sizeof(eth_stats) * k);

  key_t key_msg_q = ftok(queue_path, 'b');
  int msg_q_id = msgget(key_msg_q, 0666 | IPC_CREAT);
  if(msg_q_id < 0) {
    std::cerr << "Could not open message queue, error number " << msg_q_id << std::endl;
    return -1;
  }
  std::cout << "Opened queue with ID " << key_msg_q << std::endl;

  clock_gettime(CLOCK_REALTIME, &t_start);
  clock_gettime(CLOCK_REALTIME, &t_now);
  while(t_now.tv_sec - t_start.tv_sec < 10) {
    clock_gettime(CLOCK_REALTIME, &t_now);
    if (msgrcv(msg_q_id, stats, sizeof(eth_stats) * k, abs_node_num, 0) == -1) {
      std::cerr << "Error receiving message." << std::endl;
    } else {
      if(DEBUG>0){std::cout << std::endl << "Received message with content" << std::endl;}
      for (int j = 0; j < k; j++) { 
        if(DEBUG>0){print_stat(stats + j); }}
    }
    sleep(1);
  }
  free(stats);
  return 0;
}


int send_update_from_sysv(int k, char * node, int node_idx, char * mcast_group, char * work_dir) {
  struct timespec start{}, tv{};
  // if(node == "tor0"){
    struct timespec t_s1{}, t_s2{}, t_n1{}, t_n2{};
    unsigned long long int t_send, t_nn;
    std::ofstream myfile;
    std::cout << "NODE NAME: " << node << std::endl;
    char fn_start[100] = "/home/sim/dc-mininet-emulation/time_tracking_logs/node_";
    char node_name[15];
    strcpy(node_name,node);
    strcat(fn_start, node_name);
    // string file_name = "time_tracking_node_" + node_name + ".log";
    myfile.open (strcat(fn_start,".log"));
  // }
  struct sockaddr_in address{};
  int num_bins = 1;
  int bin_size = 64;
  char queue_path[256];
  make_queue_path(queue_path, work_dir);
  std::cout << "Send messages to " << queue_path << std::endl;
  // Initialize the interface stat structs.
  auto stats = (eth_stats *) malloc(sizeof(eth_stats) * k);

  // ---------------------------- Open the Queue ----------------------------------------
  key_t key_msg_q = ftok(queue_path, 'b');
  int msg_q_id = msgget(key_msg_q, 0666);
  if(msg_q_id < 0) {
    std::cerr << "Could not open message queue, error number " << errno << std::endl;
    return -1;
  }
  std::cout << "Opened queue with ID " << key_msg_q << std::endl;

  // -------------------- Load the Neural Network. ---------------------------------------
  // std::string model_path = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  char model_path[256];
  strcpy(model_path, work_dir);
  strcpy(model_path + strlen(work_dir), "/cpp");
  strcpy(model_path + strlen(work_dir) + strlen("/cpp"), "/traced-hlsa-module.pt");
  // std::string model_path = "/home/mininet/dc-emulation/cpp/traced-hlsa-module.pt";
  torch::jit::script::Module module;
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  auto hnsa = (uint64_t*)malloc(num_bins * sizeof(u_int64_t));
  auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  torch::Tensor network_states = torch::zeros({1, 1, 8, 6}, options);
  torch::Tensor network_masks = torch::ones({1, 1, 8, 1}, options);
  at::Tensor output;
  inputs.emplace_back(network_states);
  // inputs.emplace_back(network_masks);

  try {
    torch::NoGradGuard no_grad;
    module = torch::jit::load(model_path);
    std::cout << "Loaded model" << std::endl;

    // ------------------------ Initialize the multicast ---------------------------------
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    // char group[] = "239.0.0.1"; // argv[1];
    int port = 5007;  // atoi(argv[2]);
    int ret;
    if (sock < 0) {
      std::cerr << "Error opening socket" << std::endl;
      exit(EXIT_FAILURE);
    }
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(mcast_group);
    address.sin_port = htons(port);
    std::cout << "Configured interface to send multicast messages" << std::endl;

    // --------------------------- Start the main loop that receives and sends HNSAs -------------------------
    // Start the loop that fetches new interface statistics, executes the NN and
    // sends the update message.
    clock_gettime(CLOCK_REALTIME, &start);
    do {
      clock_gettime(CLOCK_REALTIME, &tv);
      // Fetch a new message from the queue. This call is blocking, i.e., process waits until a new message is in the
      // queue.
      if (msgrcv(msg_q_id, stats, sizeof(eth_stats) * k, node_idx, 0) > 0) {
        // VON HIER
        // if(node == "tor0"){
        clock_gettime(CLOCK_REALTIME, &t_s1); //}
        for (int i = 0; i < k; i++) {
          network_states[0][0][i][0] = (float) (stats[i].up_receive);
          network_states[0][0][i][1] = 1 - (float) (stats[i].up_receive);
          network_states[0][0][i][2] = (float) (stats[i].up_transmit);
          network_states[0][0][i][3] = 1 - (float) (stats[i].up_transmit);
          network_states[0][0][i][4] = stats[i].rate_transmitted; // currently this is already the utilization
          network_states[0][0][i][5] = stats[i].rate_received; // currently this is already the utilization
          if(DEBUG>0){
            std::cout << " Interface features of " << stats[i].iface_name << ": ";  
            print_stat(stats + i);
          }
        }
        if(DEBUG>0){std::cout << "Network States: " << network_states << std::endl;}
        // VON HIER
        // if(node == "tor0"){
          clock_gettime(CLOCK_REALTIME, &t_n1);//}
        output = module.forward(inputs).toTensor();
        // BIS HIER
        // if(node == "tor0"){
          clock_gettime(CLOCK_REALTIME, &t_n2); //}
        if(DEBUG>0){
          std::cout << "The outut is: ";
          for (int i = 0; i < 128; i++) {
            if (i % 2 == 0) std::cout << " ";
            std::cout << output[0][0][i].item<float>();
          }
          std::cout << std::endl;
        }
        make_hnsa_content(output, hnsa, num_bins, bin_size);
        if(DEBUG>0){
          for (int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1) << (1 - ((hnsa[0] >> i) & 1));
          std::cout << std::endl;
          for (int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1);
          std::cout << std::endl;

          // uint32_t tmp;
          // float elapsed = (float) (tv.tv_sec - start.tv_sec);
          // char message[] = "message from 239.0.0.1";
          // memcpy(&tmp, &elapsed, sizeof(tmp));
          // tmp = htonl(tmp);
          // ret = sendto(sock, message, strlen(message), 0, (struct sockaddr *)&address, sizeof(address));
          std::cout << "Send the integer " << hnsa[0] << std::endl;
        }
        ret = sendto(sock, hnsa, sizeof(uint64_t) * num_bins, 0, (struct sockaddr *) &address, sizeof(address));
        // BIS HIER
        // if(node == "tor0"){
          clock_gettime(CLOCK_REALTIME, &t_s2);
          t_send = ((t_s2.tv_sec - t_s1.tv_sec) * 1000000000UL + (t_s2.tv_nsec - t_s1.tv_nsec));
          t_nn = ((t_n2.tv_sec - t_n1.tv_sec) * 1000000000UL + (t_n2.tv_nsec - t_n1.tv_nsec));
          myfile << "Time of sending: " << t_send << "  ";
          myfile << "Time of NN: " << t_nn << std::endl;
        // }
        // UL stands for unsigned long and makes sure, the 1e9 is read correctly
	      // ret = 1;
        if (ret < 0) {
          std::cerr << "Error sending multicast message" << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    } while (tv.tv_sec - start.tv_sec < 12*3600);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  free(hnsa);
  free(stats);
  // if(node == "tor0"){
    myfile.close(); //}
  return EXIT_SUCCESS;
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


int directly_send_update_from_sysv(int k, char * node, int node_idx, char * mcast_group, char * work_dir) {
  struct timespec start{}, tv{};
  // if(node == "tor0"){
  struct timespec t_s1{}, t_s2{}, t_n1{}, t_n2{};
  unsigned long long int t_send, t_nn;
  std::ofstream myfile;
  std::cout << "NODE NAME: " << node << std::endl;
  char fn_start[100] = "/home/sim/dc-mininet-emulation/time_tracking_logs/node_";
  char node_name[15];
  strcpy(node_name,node);
  strcat(fn_start, node_name);
  // string file_name = "time_tracking_node_" + node_name + ".log";
  myfile.open (strcat(fn_start,".log"));
  // }
  struct sockaddr_in address{};
  int num_bins = 1;
  int bin_size = 64;
  char queue_path[256];
  make_queue_path(queue_path, work_dir);
  std::cout << "Send messages to " << queue_path << std::endl;
  // Initialize the interface stat structs.
  auto stats = (eth_stats *) malloc(sizeof(eth_stats) * k);

  // ---------------------------- Open the Queue ----------------------------------------
  key_t key_msg_q = ftok(queue_path, 'b');
  int msg_q_id = msgget(key_msg_q, 0666);
  if(msg_q_id < 0) {
    std::cerr << "Could not open message queue, error number " << errno << std::endl;
    return -1;
  }
  std::cout << "Opened queue with ID " << key_msg_q << std::endl;

  // -------------------- Load the Neural Network. ---------------------------------------
  // std::string model_path = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/traced-hlsa-module.pt";
  char model_path[256] = "/home/sim/dc-mininet-emulation/cpp/parameters-hlsa-module-stride-1.bin";;
  // strcpy(model_path, work_dir);
  // strcpy(model_path + strlen(work_dir), "/cpp");
  // strcpy(model_path + strlen(work_dir) + strlen("/cpp"), "/traced-hlsa-module.pt");

  auto hnsa = (uint64_t*)malloc(num_bins * sizeof(u_int64_t));
  int numNnParams = 48 * 128 + 128;
  auto nnParams = (float*)calloc(numNnParams, sizeof(float));
  auto prediction = (float*)calloc(128, sizeof(float));
  if(readParams(model_path, nnParams, numNnParams) != 0) {
    std::cerr << "Could not read NN parameters!" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    // ------------------------ Initialize the multicast ---------------------------------
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    // char group[] = "239.0.0.1"; // argv[1];
    int port = 5007;  // atoi(argv[2]);
    int ret;
    if (sock < 0) {
      std::cerr << "Error opening socket" << std::endl;
      exit(EXIT_FAILURE);
    }
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(mcast_group);
    address.sin_port = htons(port);
    std::cout << "Configured interface to send multicast messages" << std::endl;

    // --------------------------- Start the main loop that receives and sends HNSAs -------------------------
    // Start the loop that fetches new interface statistics, executes the NN and
    // sends the update message.
    clock_gettime(CLOCK_REALTIME, &start);
    do {
      clock_gettime(CLOCK_REALTIME, &tv);
      // Fetch a new message from the queue. This call is blocking, i.e., process waits until a new message is in the
      // queue.
      if (msgrcv(msg_q_id, stats, sizeof(eth_stats) * k, node_idx, 0) > 0) {
        // VON HIER
        clock_gettime(CLOCK_REALTIME, &t_s1); //}
        // if(node == "tor0"){
        //directGemmAvx(nnParams, stats, prediction, 1, 48, 48, 128);
        clock_gettime(CLOCK_REALTIME, &t_n1);//}
        directGemm(nnParams, stats, prediction, 1, 48, 128);
        direct_make_hnsa_content(prediction, hnsa, num_bins, bin_size);
        clock_gettime(CLOCK_REALTIME, &t_n2); //}
        // BIS HIER
        // if(node == "tor0"){
        if(DEBUG>0){
          for (int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1) << (1 - ((hnsa[0] >> i) & 1));
          std::cout << std::endl;
          for (int i = 0; i < 64; i++) std::cout << " " << ((hnsa[0] >> i) & 1);
          std::cout << std::endl;

          // uint32_t tmp;
          // float elapsed = (float) (tv.tv_sec - start.tv_sec);
          // char message[] = "message from 239.0.0.1";
          // memcpy(&tmp, &elapsed, sizeof(tmp));
          // tmp = htonl(tmp);
          // ret = sendto(sock, message, strlen(message), 0, (struct sockaddr *)&address, sizeof(address));
          std::cout << "Send the integer " << hnsa[0] << std::endl;
        }
        ret = sendto(sock, hnsa, sizeof(uint64_t) * num_bins, 0, (struct sockaddr *) &address, sizeof(address));
        // BIS HIER
        // if(node == "tor0"){
        clock_gettime(CLOCK_REALTIME, &t_s2);
        t_send = ((t_s2.tv_sec - t_s1.tv_sec) * 1000000000UL + (t_s2.tv_nsec - t_s1.tv_nsec));
        t_nn = ((t_n2.tv_sec - t_n1.tv_sec) * 1000000000UL + (t_n2.tv_nsec - t_n1.tv_nsec));
        myfile << "Time of sending: " << t_send << "  ";
        myfile << "Time of NN: " << t_nn << std::endl;
        // }
        // UL stands for unsigned long and makes sure, the 1e9 is read correctly
        // ret = 1;
        if (ret < 0) {
          std::cerr << "Error sending multicast message" << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    } while (tv.tv_sec - start.tv_sec < 12*3600);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  free(hnsa);
  free(stats);
  free(prediction);
  free(nnParams);
  // if(node == "tor0"){
  myfile.close(); //}
  return EXIT_SUCCESS;
}


int testDirectlySendUpdate() {
  auto stats = (struct eth_stats*)malloc(8 * sizeof(struct eth_stats));
  struct eth_stats * stat;
  char node[] = "tor0";
  int num_bins = 1;
  int bin_size = 64;
  uint64_t targets[] = {5720976303843509535, 6770314467273409815};
  init_node(stats, 8, node);
  // Link Failure
  // 1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   0, 1,   0, 1,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   0, 1,   0, 1,   0, 1,   0, 1,   0, 1,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,   0, 1,   1, 0,   1, 0,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   1, 0,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,   0, 1,   1, 0,   1, 0,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,
  // 5720976303843509535
  //
  // No link failure
  // 1, 0,   1, 0,   1, 0,   0, 1,   1, 0,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   0, 1,   0, 1,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   0, 1,   0, 1,   0, 1,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,   0, 1,   1, 0,   1, 0,   0, 1,   0, 1,   0, 1,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   0, 1,   1, 0,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   1, 0,   0, 1,   1, 0,   1, 0,   1, 0,   0, 1,   1, 0,   0, 1,
  // 6770314467273409815
  // char model_path[] = "/home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp/parameters-hlsa-module-stride-1.bin";
  char model_path[] = "/home/sim/dc-mininet-emulation/cpp/parameters-hlsa-module-stride-1.bin";

  auto hnsa = (uint64_t*)malloc(num_bins * sizeof(u_int64_t));
  int numNnParams = 48 * 128 + 128;
  auto nnParams = (float*)calloc(numNnParams, sizeof(float));
  auto prediction = (float*)calloc(128, sizeof(float));
  if(readParams(model_path, nnParams, numNnParams) != 0) {
    std::cerr << "Could not read NN parameters!" << std::endl;
    return EXIT_FAILURE;
  }

  bool available;
  int retCode = EXIT_SUCCESS;
  for (int j = 0; j < 2; j++) {
    for(int i = 0; i < 8; i++) {
      available = !(j == 0 && i == 4);
      stat = stats + i;
      stat->up_receive = available;
      stat->up_transmit = available;
      if (!available) {
        set_fix_stats(stat, -1);
      } else if (strcmp(stat->iface_name, "tor0-eth1") == 0) {
        set_fix_stats(stat, 0.05);
      } else if (strcmp(stat->iface_name, "tor0-eth2") == 0) {
        set_fix_stats(stat, 0.20);
      } else if (strcmp(stat->iface_name, "tor0-eth3") == 0) {
        set_fix_stats(stat, 0.15);
      } else if (strcmp(stat->iface_name, "tor0-eth4") == 0) {
        set_fix_stats(stat, 0.10);
      } else if (strcmp(stat->iface_name, "tor0-eth5") == 0) {
        set_fix_stats(stat, 0.001);
      } else if (strcmp(stat->iface_name, "tor0-eth6") == 0) {
        set_fix_stats(stat, 0.50);
      } else if (strcmp(stat->iface_name, "tor0-eth7") == 0) {
        set_fix_stats(stat, 0.85);
      } else if (strcmp(stat->iface_name, "tor0-eth8") == 0) {
        set_fix_stats(stat, 0.90);
      }
    }
    //directGemmAvx(nnParams, stats, prediction, 1, 48, 48, 128);
    uint64_t s = get_nsecs();
    directGemm(nnParams, stats, prediction, 1, 48, 128);
    std::cout << "Execution took " << (double)(get_nsecs() - s) / 1000000. << "ms" << std::endl;
    // relu(prediction, 128);
    direct_make_hnsa_content(prediction, hnsa, num_bins, bin_size);
    std::cout << "HNSA for availability " << j << " is " << hnsa[0] << std::endl;
    if(hnsa[0] != targets[j]) {
      std::cerr << "Expected HNSA to be " << targets[j] << " got " << prediction[0] << " instead for j = " << j << std::endl;
      retCode = EXIT_FAILURE;
      break;
    }
  }
  free(hnsa);
  free(nnParams);
  free(stats);
  return retCode;
}


void print_help() {
  std::cerr << "Usage: " << std::endl << "\t- ./makeHnsa producer <fat-tree-k> <logdir> <max-num-pods> <duration>" << std::endl
            << "\t- ./makeHnsa consumer <fat-tree-k> <logdir> <node-name> <node-index> <multi-cast-group>" << std::endl
            << "\t- ./makeHnsa initstate a b c d" << std::endl;
}


int main (int argc, char *argv[]) {
  // char interface[] = "wlp3s0";
  // monitor_one_interface(interface);
  // char node[] = "tor0";
  //monitor_one_node(node, 4);
  // send_updates(node, 4);
  // test_nn();
  //test_vector_assignment();

  // return testDirectlySendUpdate();


  /* Arguments are:
   * - The degree k of the fat-tree.
   * - The name of the node.
   * - The index of the node.
   */
  char * p;
  if(argc < 4) {
    std::cerr << "Too few arguments." << std::endl;
    print_help();
    return EXIT_FAILURE;
  }

  char qp[256];
  make_queue_path(qp, argv[3]);
  std::cout << "Queue located at " << qp << std::endl;

  int k = (int)strtol(argv[2], &p, 10);
  char * work_dir = argv[3];

  if(strcmp(argv[1], "producer") == 0) {
    int max_num_pods = (int)strtol(argv[4], &p, 10);
    int duration = (int)strtol(argv[5], &p, 10);
    std::cout << "START PRODUCER" << std::endl;
    producer(k, max_num_pods, work_dir, duration);
  } else if (strcmp(argv[1], "avxconsumer") == 0) {
    char * node = argv[4];
    int node_idx = (int)strtol(argv[5], &p, 10);
    char * group = argv[6];
    std::cout << "START CONSUMER with AVXNN" << std::endl;
    std::cout << "k: " << k << ", Node: " << node << ", node_idx: " << node_idx << ", mcast group: " << group << std::endl;
    directly_send_update_from_sysv(k, node, node_idx, group, work_dir);
  } else if (strcmp(argv[1], "consumer") == 0) {
    char * node = argv[4];
    int node_idx = (int)strtol(argv[5], &p, 10);
    char * group = argv[6];
    std::cout << "START CONSUMER" << std::endl;
    std::cout << "k: " << k << ", Node: " << node << ", node_idx: " << node_idx << ", mcast group: " << group << std::endl;
    // send_update_from_sysv(atoi(argv[1]), argv[2], atoi(argv[3]));
    send_update_from_sysv(k, node, node_idx, group, work_dir);
  } else if (strcmp(argv[1], "initstate") == 0) {
    initialize_data();
  } else {
    std::cerr << "Unknown setting " << argv[1] << std::endl;
    print_help();
  }
  return 0;
}
