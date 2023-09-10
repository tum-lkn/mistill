#include <linux/bpf.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <fcntl.h>
#include <bpf/libbpf.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <arpa/inet.h>

#define K 8 // k=8 fat tree

struct bpf_elf_map {
	__u32 type;
	__u32 size_key;
	__u32 size_value;
	__u32 max_elem;
	__u32 flags;
	__u32 id;
	__u32 pinning;
};

void convert(__u32 *v1, __u32 *v2) {
    *v1 = htonl(*v2);
}

int main() {
    char *path = (char*)"/sys/fs/bpf/tc/globals/network_state";
    union bpf_attr attr = {
        .pathname = (__u64)path,
        .bpf_fd = 0,
        .file_flags = 0
    };
    __u32 map_fd = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
    if(map_fd == -1){
        printf("Map network_state does not exist! Abort operation. \n");
        return 0;
    }
    union bpf_attr map_attr = {
        .map_fd = map_fd,
        .key = 0,
        .value = 0
    };

    int a, b, c, i = 0;
    __u32 n = 0;
    __u64 val, val2;
    for(a = 0; a < K; a++){
        for(b = 0; b < K; b++){
            c = 1;
            int key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 239;
            map_attr.key = (__u64)&key;
            map_attr.value = (__u64)&val;
            int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
            if(ret == -1){
                // printf("Reading process of map network_state failed! No information for IP: %d.%d.%d.%d. \n", 239, a, b, c);
                continue;
            }
            convert((__u32*)(&val2),(__u32*)(&val)+1);
            convert((__u32*)(&val2)+1,(__u32*)(&val));
            printf("[network state] IP: %d.%d.%d.%d information: %llu %llu \n", 239, a, b, c, val, val2);
        }
    }
    a = K;
    for(b = 1; b <= K/2; b++){
        for(c = 1; c <= K/2; c++){
            int key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 239;
            map_attr.key = (__u64)&key;
            map_attr.value = (__u64)&val;
            int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
            if(ret == -1){
                // printf("Reading process of map network_state failed! No information for IP: %d.%d.%d.%d. \n", 239, a, b, c);
                continue;
            }
            convert((__u32*)(&val2),(__u32*)(&val)+1);
            convert((__u32*)(&val2)+1,(__u32*)(&val));
            printf("[network state] IP: %d.%d.%d.%d information: %llu %llu \n", 239, a, b, c, val, val2);
        }
    }
    printf("[network state] done\n");
    return 0;
}