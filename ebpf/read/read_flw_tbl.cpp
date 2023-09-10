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

int main() {
    char *path = (char*)"/sys/fs/bpf/tc/globals/flow_table";
    union bpf_attr attr = {
        .pathname = (__u64)path,
        .bpf_fd = 0,
        .file_flags = 0
    };
    __u32 map_fd = syscall(__NR_bpf, BPF_OBJ_GET, &attr, sizeof(attr));
    if(map_fd == -1){
        printf("Map flow_table does not exist! Abort operation. \n");
        return 0;
    }
    union bpf_attr map_attr = {
        .map_fd = map_fd,
        .key = 0,
        .value = 0
    };

    __u32 n = 0;
    __u32 val[6];
    for(int a = 0; a < 8; a++){
        for(int b = 0; b < 4; b++){
            for(int c = 2; c < 6; c++){
                int key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 10;
                map_attr.key = (__u64)&key;
                for(n=0; n<6; n++){
                    val[n] = 0;
                }
                map_attr.value = (__u64)&val[0];
                int ret = syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &map_attr, sizeof(map_attr));
                if(ret == -1){
                // printf("Reading process of map flow_table failed! No labels for IP: %d.%d.%d.%d. \n", 10, a, b, c);
                continue;
                }
                printf("[mpls labels] IP: %d.%d.%d.%d labels: %d %d %d %d %d %d \n", 10, a, b, c, val[0], val[1], val[2], val[3], val[4], val[5]);
            }
        }
    }
    printf("[flow table labels] done\n");
    return 0;
}