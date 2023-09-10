#include <linux/bpf.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <fcntl.h>
#include <bpf/libbpf.h>
#include <sys/syscall.h>
#include <unistd.h>

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
    .pathname = (__u64)(char*)"/sys/fs/bpf/tc/globals/t_kernel",
    .bpf_fd = 0,
    .file_flags = 0
};
union bpf_attr map_attr = {
    .map_fd = 0,
    .key = 0,
    .value = 0
};

int main(int argc, char** argv) {
  // Read kernel time tracking map
  // arguments:
  // - name of json file, where data is saved to
  char spath[50];
  char* s_start = (char*)"test_data/";
  char* s_middle = (char*)"kernel_";
  strcat(spath, s_start);
  strcat(spath, argv[1]);
  strcat(spath,s_middle);
  strcat(spath, argv[2]);
  strcat(spath, ".json");
  printf("saving to: %s\n", spath); // debugging but good for overview
  FILE *f;
  f = fopen(spath,"w");
  fprintf(f,"{\"Data\": [\n");
  
  __u32 map_fd = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
  if(map_fd == -1){
    printf("Map t_kernel does not exist! Abort operation. \n");
    return 0;
  }
  __u64 j = 0;
  __u64 val[2];
  map_attr.map_fd = map_fd;
  map_attr.key = (__u64)&j;
  map_attr.value = (__u64)&val[0];
  int test = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(test == -1){
    printf("Reading process of map t_kernel failed! Abort operation. \n");
    fprintf(f,"]}");
    fclose(f);
    return 0;
  }
  printf("[len] of all: %lld \n", val[1]);
  int len = val[1];
  __u64 i=1;
  map_attr.key = (__u64)&i;
  map_attr.value = (__u64)&val[0];
  int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
  if(ret == -1){
    printf("Reading process of map t_user failed! Abort operation. \n");
    fprintf(f,"]}");
    fclose(f);
    return 0;
  }
  fprintf(f,"{\"flag\": %lld, \"time\": %lld}", val[0], val[1]);
  for(i=2; i<=len; i++){
    map_attr.key = (__u64)&i;
    map_attr.value = (__u64)&val[0];
    int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
    if(ret == -1){
      printf("Reading process of map t_user failed! Abort operation. \n");
      fprintf(f,"]}");
      fclose(f);
      return 0;
    }
    fprintf(f,",\n");
    fprintf(f,"{\"flag\": %lld, \"time\": %lld}", val[0], val[1]);
  }
  fprintf(f,"]}");
  fclose(f);
}
