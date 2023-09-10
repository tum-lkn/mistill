#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <net/if_arp.h>
#include <linux/icmp.h>
#include <linux/in.h>
#include <stddef.h>
#include "bpf_helpers.h"
#include "mpls.h"

/* compiler workaround */
// #define bpf_htonl __builtin_bswap32
#define bpf_memcpy __builtin_memcpy

#define MAX_IP_HDR_LEN 60
#define FLOW_TABLE_ELEMS 128 // vorerst
#define NETWORK_STATE_ELEMS 80 // vorerst
#define MAX_HOPS 6 // vorerst
#define MAP_TIME_SIZE 1800000 // 30k pps for 60s // pretty huge for one ping per second, but whatever
#define TIME_TRACK_FLAG 1 // 0 if time should not be tracked
#define K 8 // k=8 fat tree system
#define FLAG_EGRESS 1
#define FLAG_INGRESS_DATA 2
#define FLAG_INGRESS_UPDATE 3
#define FLAG_MPLS_ACTIVE 4
#define FLAG_MPLS_INACTIVE 9
#define FLAG_MPLS_UNKNOWN 14

#define PIN_GLOBAL_NS 2
typedef __u32 uINTs;
typedef __u64 uINT; // originally __u64

struct bpf_elf_map SEC("maps") nn_labels = {
	.type 			= BPF_MAP_TYPE_HASH, // allows for flexible keys
	.size_key 	    = sizeof(uINTs), // destination IP of end-host
	.size_value     = sizeof(uINTs) * MAX_HOPS, // multiple MPLS labels
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out from user space
	.max_elem 	    = FLOW_TABLE_ELEMS,
};
// struct { // potential new way of defining maps; but it does not work yet?
// 	__uint(type, BPF_MAP_TYPE_HASH); // allows for flexible keys
// 	__type(key, uINTs); // destination IP of end-host
// 	__type(value, uINTs* MAX_HOPS); // multiple MPLS labels
// 	__uint(pinning, PIN_GLOBAL_NS); // global pin to read it out from user space
// 	__uint(max_entries, FLOW_TABLE_ELEMS);
// } nn_labels SEC(".maps");
struct bpf_elf_map SEC("maps") flow_table = {
	.type 			= BPF_MAP_TYPE_HASH, // allows for flexible keys
	.size_key 	    = sizeof(uINTs), // destination IP of end-host
	.size_value     = sizeof(uINTs) * (MAX_HOPS), // multiple MPLS labels
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out from user space
	.max_elem 	    = FLOW_TABLE_ELEMS,
};
struct bpf_elf_map SEC("maps") dest_IPs = {
	.type 			= BPF_MAP_TYPE_HASH, // allows for flexible keys
	.size_key 	    = sizeof(uINTs), // destination IP of end-host
	.size_value     = sizeof(uINT), // time of last contact
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out from user space
	.max_elem 	    = (FLOW_TABLE_ELEMS),
};
struct bpf_elf_map SEC("maps") network_state = {
	.type 			= BPF_MAP_TYPE_HASH,
	.size_key 	    = sizeof(uINTs), // source IP of switch
	.size_value     = sizeof(uINT), // 64 bit update message for the NN
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out from user space
	.max_elem 	    = NETWORK_STATE_ELEMS,
};
struct bpf_elf_map SEC("maps") t_kernel = {
	.type 			= BPF_MAP_TYPE_HASH,
	.size_key 	    = sizeof(uINT),
	.size_value     = sizeof(uINT)*2, // flag and time
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out after program ends
	.max_elem 	    = MAP_TIME_SIZE,
};
struct bpf_elf_map SEC("maps") t_user = {
	.type 			= BPF_MAP_TYPE_HASH,
	.size_key 	    = sizeof(uINT),
	.size_value     = sizeof(uINT)*2, // flag and time
	.pinning 		= PIN_GLOBAL_NS, // global pin to read it out after program ends
	.max_elem 	    = MAP_TIME_SIZE,
};
#define trace_printk(fmt, ...) do { \
	char _fmt[] = fmt; \
	bpf_trace_printk(_fmt, sizeof(_fmt), ##__VA_ARGS__); \
	} while (0)

static void track_time(uINT t_diff, uINT flag){
	// functionality from research internship
	if(TIME_TRACK_FLAG==0){return;}
	uINT k = 0, v[2], k3;
	uINT *k2 = bpf_map_lookup_elem(&t_kernel, &k);
	if(k2 != NULL){
        if(*(k2+1) > MAP_TIME_SIZE){
			// trace_printk("MAP %s has reset!\n", "t_kernel");
            do{
                *(k2+1) = *(k2+1) - MAP_TIME_SIZE; // if the size limit is reached, the first values are overwritten again
            }while(*(k2+1) > MAP_TIME_SIZE); // this actually resets the map also for the program that reads it out afterwards
        }
		k3 = *(k2+1) +1;
		v[0] = flag;
		v[1] = t_diff;
		bpf_map_update_elem(&t_kernel, &k3, &v, BPF_ANY);
		v[0] = flag;
		v[1] = k3;
		bpf_map_update_elem(&t_kernel, &k, &v, BPF_ANY);
		// trace_printk("[time] Duration: %d", t_diff); // debugging
	} else { // first entry in the map
		v[0] = flag;
		v[1] = 1;
		bpf_map_update_elem(&t_kernel, &k, &v, BPF_ANY);
		k3 = 1;
		v[0] = flag;
		v[1] = t_diff;
		bpf_map_update_elem(&t_kernel, &k3, &v, BPF_ANY);
		// trace_printk("[time] %d", t_diff); // debugging
	}
}

static void init_network_state() {
	uINTs k = ( 1 << 24 ) | ( 0 << 16 ) | ( 0 << 8 ) | 239; // 239.0.0.1 is  the Multicast-IP of the first ToR switch
	uINT *test = bpf_map_lookup_elem(&network_state, &k);

	if(test == NULL){ // run this code only once, when this program is initially called
		uINT  init_values[80] = {6178267072579306743UL, 15131020779273914071UL, 15131091148051645687UL, 5905445312561742999UL, 
			15419211573040580343UL, 6195909904897476311UL, 15131016381262005463UL, 6193680095349900439UL, 15128799765819361015UL, 
			6193662503163856087UL, 15130972400763339991UL, 6193658105083791095UL, 15419281941752251607UL, 15419290729288893653UL, 
			15131033973414495479UL, 5905432127012144343UL, 15130981196889916599UL, 15131025177355027671UL, 15128768979461276887UL, 
			15419202768358671575UL, 15130985594935379095UL, 6195861526418359511UL, 5905427728964585207UL, 5905440923070564055UL, 
			15128799765819360983UL, 15131016381227402967UL, 15130985594936427735UL, 15130985594902873303UL, 5907617947504674007UL, 
			5905427728964585207UL, 15419220369134651095UL, 6193658105116296919UL, 5907626752202835573UL, 5905423330918074071UL, 
			6193578931689162389UL, 5905489293025805975UL, 6195914302976493271UL, 15128834950157895383UL, 15131038371494560983UL, 
			15131016381228451575UL, 5907609160035141335UL, 15131007585134380759UL, 15130981196889916631UL, 15419211564450645655UL, 
			15128843737694537431UL, 15419215971087090903UL, 15128839348237960407UL, 15130981196856362199UL, 6195843934233363671UL, 
			15131038371493512855UL, 15130998789075961047UL, 15131016381262005911UL, 15128839348237960343UL, 15128826154065921239UL, 
			15131086750005134999UL, 15128799765820409047UL, 6195931895163585751UL, 15131016381227402967UL, 5905432126977541847UL, 
			6193658105083791063UL, 15130998789074913015UL, 15128799765785806551UL, 5905449719197140695UL, 15128834950191449239UL, 
			5905432127011095767UL, 15419215971088139479UL, 5905427728931030775UL, 15131038371494560983UL, 15131073555866125463UL, 
			15131033973448049815UL, 6195909904929982167UL, 6195909904931030231UL, 15128834950157895383UL, 15130989992949384407UL, 
			15128843737695585495UL, 5905449719198188759UL, 15130998789074912503UL, 15131025168764045015UL, 15130998789074912983UL, 
			15130985594936427735UL};
		uINTs labels[MAX_HOPS];
		uINTs a, b, c = 1, key, i = 0;
		for(a = 0; a < K; a++){
			for(b = 0; b < K/2; b++){
				key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 239;
				uINT val = init_values[i]; // real value
				i = i + 1;
				// trace_printk("[ift] flow table key: %d\n", key);
				bpf_map_update_elem(&network_state, &key, &val, BPF_ANY);
			}
		}
		for(a = 0; a < K; a++){
			for(b = K/2; b < K; b++){
				key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 239;
				uINT val = init_values[i]; // real value
				i = i + 1;
				// trace_printk("[ift] flow table key: %d\n", key);
				bpf_map_update_elem(&network_state, &key, &val, BPF_ANY);
			}
		}
		a = K;
		for(b = 1; b <= K/2; b++){
			for(c = 1; c <= K/2; c++){
				key = ( c << 24 ) | ( b << 16 ) | ( a << 8 ) | 239;
				uINT val = init_values[i]; // real value
				i = i + 1;
				// trace_printk("[ift] flow table key: %d\n", key);
				bpf_map_update_elem(&network_state, &key, &val, BPF_ANY);
			}
		}
		trace_printk("[eBPF] network state now initialized with %d entries\n", NETWORK_STATE_ELEMS);
	} else {
		// already initialized
		// trace_printk("[eBPF] network state already initialized: %d\n", k);
	}
}

static int add_MPLS_header(uINTs dst_ip, struct __sk_buff *skb, int iph_len, __u8 ttl_ip) {
	uINT t1 = bpf_ktime_get_ns();
	/* This is the amount of padding we need to add to be able to add the mpls header */
	// trace_printk("[mpls] start adding header: %d\n", 1);
	int padlen = sizeof(struct mpls_hdr);
	uINTs labels[6];
	for(int i = 0; i < MAX_HOPS; i++){
		labels[i] = 0;
	}
	uINTs *flows = bpf_map_lookup_elem(&flow_table, &dst_ip);
	uINT *t_flow = bpf_map_lookup_elem(&dest_IPs, &dst_ip);
	// trace_printk("[eBPF] grabbed MPLS labels for dest_IP: %d\n", dst_ip);
	bool BoS = true; // bottom of stack -> only for first written label
	uINTs label = 0;
	uINT t = bpf_ktime_get_ns();
	int return_flag;

	if(flows != NULL){
		if(t_flow != NULL){
			if((t-*t_flow) > 100000000) { // 100ms // check if flow is not recent
				uINTs *labels_read = bpf_map_lookup_elem(&nn_labels, &dst_ip);
				if(labels_read != NULL){
					uINTs flow_labels[MAX_HOPS];
					for(int i = 0; i < MAX_HOPS; i++){
						labels[i] = labels_read[i];
						flow_labels[i] = labels_read[i];
					}
					return_flag = FLAG_MPLS_INACTIVE; // previously inactive flow
					// trace_printk("[time] previously inactive flow: %ld\n", t);
					bpf_map_update_elem(&flow_table, &dst_ip, &flow_labels, BPF_ANY);
				}
			} else { // if flow is recent, then the labels to-be-used were already read
				for(int i = 0; i < MAX_HOPS; i++){
					labels[i] = flows[i];
				}
				return_flag = FLAG_MPLS_ACTIVE; // active flow
				// trace_printk("[time] active flow: %ld\n", t);
			}
		}
	} else {
		uINTs *labels_read = bpf_map_lookup_elem(&nn_labels, &dst_ip);
		if(labels_read != NULL){
			uINTs flow_labels[MAX_HOPS];
			for(int i = 0; i < MAX_HOPS; i++){
				labels[i] = labels_read[i];
				flow_labels[i] = labels_read[i];
			}
			return_flag = FLAG_MPLS_UNKNOWN; // previously unknown flow
			// trace_printk("[time] previously unknown flow: %ld\n", t);
			bpf_map_update_elem(&flow_table, &dst_ip, &flow_labels, BPF_ANY);
		}
	}
	__u32 max_i = 3; // test
	// now, the labels (from which source is not important) can be added as MPLS headers to the packet
	for(uINTs i = max_i; i < MAX_HOPS; ++i){ // normal 0 to MAX_HOPS
		// extract label from array
		if(labels[MAX_HOPS-1-i] == -1){
			// those are not needed for the current destination
			// trace_printk("[mpls] unneeded label: %ld\n", (MAX_HOPS-1-i));
		} else {
			label = labels[MAX_HOPS-1-i] + 100; // needed due to eBPF constraints 
			// + 100 due to restricted Labels 0-15 // reverse order of labels
			// if(i==0 || i==1){
			// 	trace_printk("[MPLS] encode label: %d\n", label);
			// }
			// construct our deterministic mpls header
			struct mpls_hdr mpls = mpls_encode(label, ttl_ip, 0, BoS); 
			// bool is true for bottom of stack
			/* Grow the room for data in the packet */
			// int ret = bpf_skb_adjust_room(skb, padlen, BPF_ADJ_ROOM_NET, 0);
			int ret = bpf_skb_adjust_room(skb, padlen, BPF_ADJ_ROOM_MAC, 0);
			if (ret)
				return 10;

			// unsigned long offset = sizeof(struct ethhdr) + (unsigned long)iph_len;
			unsigned long offset = sizeof(struct ethhdr);
			ret = bpf_skb_store_bytes(skb, (int)offset, &mpls, sizeof(struct mpls_hdr),
										BPF_F_RECOMPUTE_CSUM);
			BoS = false; // each following label is not BoS
		}
	} 
	// now, the time is written to the destination (happens for every packet/ function call)
	bpf_map_update_elem(&dest_IPs, &dst_ip, &t, BPF_ANY);
	// trace_printk("[time] written to IP: %ld\n", t);
	uINT t2 = bpf_ktime_get_ns();
	track_time((t2-t1),(uINT)return_flag);
    return TC_ACT_OK;
}

static int handle_update_message(struct __sk_buff *skb, struct iphdr *ip) {
	init_network_state();
	uINTs src_ip = ip->daddr;
	uINTs net_multicast = src_ip & 0xff; // A.x.x.x
	if(net_multicast == 239){ // == 239 is correct
		uINT offset = sizeof(struct ethhdr) + sizeof(struct iphdr) + 8; // = 42 // 8 is size of UDP header
		void *data_start = (void *)(long)skb->data;
		void *data_end = (void *)(long)skb->data_end;
		void *data = (void*)(data_start + offset);
		if(data + sizeof(uINT) >= data_end){
			return -1;
		}
		uINT* dat_star = (uINT*)data;
		uINT dat = *dat_star;
		bpf_map_update_elem(&network_state, &src_ip, &dat, BPF_ANY);
		// Extract octetts in the reverse order they are stored in the integer.
		// trace_printk("IP part I: 	%d.",	(src_ip) 		& 0xff); // A.x.x.x
		// trace_printk("IP part II: 	%d.", 	(src_ip >> 8) 	& 0xff); // x.A.x.x
		// trace_printk("IP part III: 	%d.", 	(src_ip >> 16) 	& 0xff); // x.x.A.x
		// trace_printk("IP part IV: 	%d",	(src_ip >> 24) 	& 0xff); // x.x.x.A
 		// trace_printk("Update Message written into Map for IP: %d", src_ip);
		return 1; // 1 to drop the update messages - kernel does not need them
	} else {
		return 0;
	}
}

SEC("ingress")
int mpls_ingress(struct __sk_buff *skb) {	
	uINT t1 = bpf_ktime_get_ns();
	/* We will access all data through pointers to structs */
	void *data = (void *)(long)skb->data;
	void *data_end = (void *)(long)skb->data_end;

	/* first we check that the packet has enough data, so we can
	 * access the three different headers of ethernet, ip and mpls  */
	if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end){
		// trace_printk("[info-i] headers longer than packet: %d\n", 1);
		return TC_ACT_SHOT;
	}

	struct ethhdr  *eth  = data;
	bool is_mpls = false;
	/* Only actual IP packets are allowed */
	if (eth->h_proto != __constant_htons(ETH_P_IP)){
		if(eth->h_proto == __constant_htons(ETH_P_MPLS_UC)){
			// trace_printk("[info] an mpls packet: %d\n", 1);
			is_mpls = true;
		} else {
		return TC_ACT_UNSPEC;
		}
	}

	struct iphdr   *ip   = (data + sizeof(struct ethhdr));
    // trace_printk("[IP] sending IP: %d\n", ip->saddr);
    /* get the number of bytes in the IP header */
    int iph_len = ip->ihl << 2;
    if (iph_len > MAX_IP_HDR_LEN) {
		// trace_printk("[info-i] IP header longer than maximum: %d\n", 1);
		return TC_ACT_SHOT;
	}
	uINT own_IP = ( 12 << 24 ) | ( 0 << 16 ) | ( 0 << 8 ) | 10;
	if(ip->saddr == own_IP){
		is_mpls = false; // if for some reason the outgoing packets appear here, we don't want to delete their labels
		// trace_printk("[info] outgoing mpls packet intercepted! %d\n", 1);
	}

    if(!is_mpls){
		int ret = handle_update_message(skb, ip);
		if(ret){
			// trace_printk("[info-i] update message: %d\n", 1);
			uINT t2 = bpf_ktime_get_ns();
			track_time((t2-t1),FLAG_INGRESS_UPDATE);
			return TC_ACT_SHOT;
		}
	}

	// UPDATE: we now assume, that the mpls header is completely removed at the last ToR switch.
	// Therefore, the following is just a failsafe to be able to work with incoming MPLS packets
	if(is_mpls){
		bool is_BoS = false;
		bool is_pkt_rdy = false;
		for(uINTs i = 0; i <= MAX_HOPS; i++){ // more than MAX_HOPS MPLS headers
			// are not possible -- additionally eBPF verifier: bounded loop
			if(is_BoS){
				if(is_pkt_rdy){
					// after the bottom of stack is reached nothing has to be done anymore
				} else {
					/* This is the amount of padding we need to remove to be just left
					* with eth * iphdr. */
					int padlen = i * sizeof(struct mpls_hdr);

					/* Shrink the room for data in the packet */
					int ret = bpf_skb_adjust_room(skb, -padlen, BPF_ADJ_ROOM_NET, 0);
					if (ret)
						return TC_ACT_SHOT;
					// trace_printk("[MPLS] removed header of length: %x\n", padlen);
					is_pkt_rdy = true;
				}
			} else {
				/* ==[ETH HDR][IP HDR][MPLS HDR][Data]== */
				// struct mpls_hdr *mpls = (struct mpls_hdr *)((void *)(ip) + iph_len);
				struct mpls_hdr *mpls = (data + sizeof(struct ethhdr) + sizeof(struct iphdr))
										+ i * sizeof(struct mpls_hdr);

				if ((void *)(mpls + 1) > data_end)
					return TC_ACT_SHOT;

				struct mpls_entry_decoded mpls_decoded = mpls_entry_decode(mpls);
				// if(i==0 || i==1){
				// 	trace_printk("[MPLS] decoded label: %d\n", mpls_decoded.label);
				// }
				if (is_mpls_entry_bos(mpls))
					is_BoS = true;
			}
		}
	}
	uINT t2 = bpf_ktime_get_ns();
	track_time((t2-t1),FLAG_INGRESS_DATA);
    return TC_ACT_OK;
}

SEC("egress")
int mpls_egress(struct __sk_buff *skb) {
	uINT t1 = bpf_ktime_get_ns();
	/* We will access all data through pointers to structs */
	void *data = (void *)(long)skb->data;
	void *data_end = (void *)(long)skb->data_end;

	/* first we check that the packet has enough data, so we can
	 * access the three different headers of ethernet, ip and mpls  */
	if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end){
		// trace_printk("[info-e] headers longer than packet: %d\n", 1);
		return TC_ACT_SHOT;
	}

	struct ethhdr  *eth  = data;
	/* Only actual IP packets are allowed */
	if (eth->h_proto != __constant_htons(ETH_P_IP)){
		// trace_printk("[info-e] not an IP packet: %d\n", 1);
		return TC_ACT_OK;
	}

	struct iphdr   *ip   = (data + sizeof(struct ethhdr));
    /* get the number of bytes in the IP header */
    int iph_len = ip->ihl << 2;
    if (iph_len > MAX_IP_HDR_LEN) {
		// trace_printk("[info-e] IP header longer than maximum: %d\n", 1);
		return TC_ACT_SHOT;
	}

	/* extract the destination ip and save it to map flow_table */
	uINTs dst_ip = ip->daddr;
	__u8 ttl_ip = ip->ttl;
	uINTs ret = add_MPLS_header(dst_ip, skb, iph_len, ttl_ip);
	if(ret==10)
		return TC_ACT_SHOT;
	__u8 eth_type1 = 0x88; //ETH_P_MPLS_UC; // 0x8847
	bpf_skb_store_bytes(skb, 12, &eth_type1, sizeof(eth_type1), BPF_F_RECOMPUTE_CSUM); // ETH Type is 13th and 14th Byte
	__u8 eth_type2 = 0x47; //ETH_P_MPLS_UC; // 0x8847
	bpf_skb_store_bytes(skb, 13, &eth_type2, sizeof(eth_type2), BPF_F_RECOMPUTE_CSUM); // ETH Type is 13th and 14th Byte

	uINT t2 = bpf_ktime_get_ns();
	track_time((t2-t1),FLAG_EGRESS);
	return TC_ACT_OK;
}

char __license[] SEC("license") = "GPL";