/*
 This is the tool ....
 Contact Author: Jian Tang, Microsoft Research, jiatang@microsoft.com, tangjianpku@gmail.com
 Publication: Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei. "LINE: Large-scale Information Network Embedding". In WWW 2015.
 */

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <boost/thread.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
//#include <pthread.h>
//#include <gsl/gsl_rng.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <map>
#include <time.h>
#if !defined(__SUNPRO_CC) || (__SUNPRO_CC > 0x530)
#include <boost/generator_iterator.hpp>
#endif



#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
    double degree;
    char *name;
};

int num_location = 8227;
char location_name[27109][MAX_STRING];
char network_file_1[MAX_STRING], network_file_2[MAX_STRING], network_file_3[MAX_STRING], embedding_file[MAX_STRING],embedding_location_file[MAX_STRING];
struct ClassVertex *vertexTotal, *vertex_1, *vertex_2, *vertex_3;
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
int *vertex_hash_table_1, *vertex_hash_table_2, *vertex_hash_table_3, *neg_table_1, *neg_table_2, *neg_table_3;
int max_num_vertices_1 = 1000, max_num_vertices_2 = 1000, max_num_vertices_3 = 1000, num_vertices_1 = 0, num_vertices_2 = 0, num_vertices_3 = 0, user_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges_1 = 0, num_edges_2 = 0, num_edges_3 = 0;
real init_rho = 0.025, rho;
real *emb_vertex_1, *emb_context_1, *emb_vertex_2, *emb_context_2, *emb_vertex_3, *emb_context_3, *emb_location, *sigmoid_table;

int *edge_source_id_1, *edge_target_id_1, *edge_source_id_2, *edge_target_id_2, *edge_source_id_3, *edge_target_id_3;
double *edge_weight_1, *edge_weight_2, *edge_weight_3;

// Parameters for edge sampling
long long *alias_1, *alias_2, *alias_3;
double *prob_1, *prob_2, *prob_3;

//random generator 
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);
//const gsl_rng_type * gsl_T;
//gsl_rng * gsl_r;

/* Build a hash table, mapping each vertex name to a unique vertex id */

int binarySearch(char* x){  
    int low, high, mid;  
	int n = num_location;
    low = 0;  
    high = n - 1;
    while ( low <= high ) {  
        mid = (low + high) / 2;  
        if(strcmp(x,location_name[mid]) < 0){  
            high = mid - 1;  
        }  
        else if(strcmp(x,location_name[mid]) > 0){  
            low = mid + 1;  
        }  
        else{  
            return mid;  
        }  
    }   
    return -1;  
}

unsigned int Hash(char *key)
{
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key)
    {
        hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
}

void InitHashTable_1()
{
    vertex_hash_table_1 = (int *)malloc(hash_table_size * sizeof(int)); //  hash_table_size=30000000
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table_1[k] = -1;
}


void InitHashTable_2()
{
    vertex_hash_table_2 = (int *)malloc(hash_table_size * sizeof(int)); //  hash_table_size=30000000
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table_2[k] = -1;
}

void InitHashTable_3()
{
    vertex_hash_table_3 = (int *)malloc(hash_table_size * sizeof(int)); //  hash_table_size=30000000
    for (int k = 0; k != hash_table_size; k++) vertex_hash_table_3[k] = -1;
}

void InsertHashTable_1(char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table_1[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table_1[addr] = value;
}

void InsertHashTable_2(char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table_2[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table_2[addr] = value;
}

void InsertHashTable_3(char *key, int value)
{
    int addr = Hash(key);
    while (vertex_hash_table_3[addr] != -1) addr = (addr + 1) % hash_table_size;
    vertex_hash_table_3[addr] = value;
}

int SearchHashTable_1(char *key)
{
    int addr = Hash(key);
    while (1)
    {
        if (vertex_hash_table_1[addr] == -1) return -1;
        if (!strcmp(key, vertex_1[vertex_hash_table_1[addr]].name)) return vertex_hash_table_1[addr];
        addr = (addr + 1) % hash_table_size;
    }
    return -1;
}

int SearchHashTable_2(char *key)
{
    int addr = Hash(key);
    int c = 0;
    while (1)
    {
        if (vertex_hash_table_2[addr] == -1) return -1;
        //printf("%d** \n", vertex_hash_table_2[addr]);x
        if (!strcmp(key, vertex_2[vertex_hash_table_2[addr]].name)) return vertex_hash_table_2[addr];
        addr = (addr + 1) % hash_table_size;
        c++;
    }
    return -1;
}

int SearchHashTable_3(char *key)
{
    int addr = Hash(key);
    int c = 0;
    while (1)
    {
        if (vertex_hash_table_3[addr] == -1) return -1;
        if (!strcmp(key, vertex_3[vertex_hash_table_3[addr]].name)) return vertex_hash_table_3[addr];
        addr = (addr + 1) % hash_table_size;
        c++;
    }
    return -1;
}

/* Add a vertex to the vertex set */
int AddVertex_1(char *name)
{
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex_1[num_vertices_1].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex_1[num_vertices_1].name, name);
    vertex_1[num_vertices_1].degree = 0;
    num_vertices_1++;
    if (num_vertices_1 + 2 >= max_num_vertices_1)
    {
        max_num_vertices_1 += 1000;
        vertex_1 = (struct ClassVertex *)realloc(vertex_1, max_num_vertices_1 * sizeof(struct ClassVertex));
    }
    InsertHashTable_1(name, num_vertices_1 - 1);
    return num_vertices_1 - 1;

}

/* Add a vertex to the vertex set */
int AddVertex_2(char *name)
{
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex_2[num_vertices_2].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex_2[num_vertices_2].name, name);
    vertex_2[num_vertices_2].degree = 0;
    num_vertices_2++;
    if (num_vertices_2 + 2 >= max_num_vertices_2)
    {
        max_num_vertices_2 += 1000;
        vertex_2 = (struct ClassVertex *)realloc(vertex_2, max_num_vertices_2 * sizeof(struct ClassVertex));
    }
    InsertHashTable_2(name, num_vertices_2 - 1);
    return num_vertices_2 - 1;
}


/* Add a vertex to the vertex set */
int AddVertex_3(char *name)
{
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex_3[num_vertices_3].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex_3[num_vertices_3].name, name);
    vertex_3[num_vertices_3].degree = 0;
    num_vertices_3++;
    if (num_vertices_3 + 2 >= max_num_vertices_3)
    {
        max_num_vertices_3 += 1000;
        vertex_3 = (struct ClassVertex *)realloc(vertex_3, max_num_vertices_3 * sizeof(struct ClassVertex));
    }
    InsertHashTable_3(name, num_vertices_3 - 1);
    return num_vertices_3 - 1;
}



/* Read network from the training file */
void ReadData()
{
    //******************************************************************************
    //reading network_file_1
    
    FILE *fin_1;
    char name_v1_1[MAX_STRING], name_v2_1[MAX_STRING], str_1[2 * MAX_STRING + 10000];
    int vid_1;
    double weight_1;
    
    fin_1 = fopen(network_file_1, "rb");
    if (fin_1 == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges_1 = 0;
    while (fgets(str_1, sizeof(str_1), fin_1)) num_edges_1++;
   // fclose(fin_1);
    printf("#edges of network_file_1: %lld          \n", num_edges_1);
    
    edge_source_id_1 = (int *)malloc(num_edges_1*sizeof(int));
    edge_target_id_1 = (int *)malloc(num_edges_1*sizeof(int));
    edge_weight_1 = (double *)malloc(num_edges_1*sizeof(double));
    if (edge_source_id_1 == NULL || edge_target_id_1 == NULL || edge_weight_1 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    fin_1 = fopen(network_file_1, "rb");
    num_vertices_1 = 0;
    for (int k = 0; k != num_edges_1; k++)
    {
        fscanf(fin_1, "%s %s %lf", name_v1_1, name_v2_1, &weight_1);
        
        if (k % 10000 == 0)
        {
            printf("Reading edges of network_file_1: %.3lf%%%c", k / (double)(num_edges_1 + 1) * 100, 13);
            fflush(stdout);
        }
        
        vid_1 = SearchHashTable_1(name_v1_1);
		if (vid_1 == -1) {
			
			vid_1 = AddVertex_1(name_v1_1);
		}
        vertex_1[vid_1].degree += weight_1;
        edge_source_id_1[k] = vid_1;
        
        vid_1 = SearchHashTable_1(name_v2_1);
		if (vid_1 == -1){
			
			vid_1 = AddVertex_1(name_v2_1);
		}
        vertex_1[vid_1].degree += weight_1;
        edge_target_id_1[k] = vid_1;
        
        edge_weight_1[k] = weight_1;
    }
    fclose(fin_1);
    printf("#vertices of network_file_1: %d          \n", num_vertices_1);
    
    
    
    //**********************************
    //reading network_file_2
    FILE *fin_2;
    char name_v1_2[MAX_STRING], name_v2_2[MAX_STRING], str_2[2 * MAX_STRING + 10000];
    int vid_2;
    double weight_2;
    fin_2 = fopen(network_file_2, "rb");
    if (fin_2 == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges_2 = 0;
    while (fgets(str_2, sizeof(str_2), fin_2)) num_edges_2++;
    //fclose(fin_2);
    printf("#edges of network_file_2: %lld          \n", num_edges_2);
    
    edge_source_id_2 = (int *)malloc(num_edges_2*sizeof(int));
    edge_target_id_2 = (int *)malloc(num_edges_2*sizeof(int));
    edge_weight_2 = (double *)malloc(num_edges_2*sizeof(double));
    if (edge_source_id_2 == NULL || edge_target_id_2 == NULL || edge_weight_2 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    fin_2 = fopen(network_file_2, "rb");
    num_vertices_2 = 0;
    for (int k = 0; k != num_edges_2; k++)
    {
        fscanf(fin_2, "%s %s %lf", name_v1_2, name_v2_2, &weight_2);
        
        if (k % 10000 == 0)
        {
            printf("Reading edges of network_file_2: %.3lf%%%c", k / (double)(num_edges_2 + 1) * 100, 13);
            fflush(stdout);
        }
        
        vid_2 = SearchHashTable_2(name_v1_2);
		if (vid_2 == -1) {
			vid_2 = AddVertex_2(name_v1_2);
		}
        vertex_2[vid_2].degree += weight_2;
        edge_source_id_2[k] = vid_2;
        
        vid_2 = SearchHashTable_2(name_v2_2);
		if (vid_2 == -1) {
			vid_2 = AddVertex_2(name_v2_2);
		}
        vertex_2[vid_2].degree += weight_2;
        edge_target_id_2[k] = vid_2;
        
        edge_weight_2[k] = weight_2;
    }
    fclose(fin_2);
    printf("#vertices of network_file_2: %d          \n", num_vertices_2);


    
    //******************************************************************************
    //reading network_file_3
    
    FILE *fin_3;
    char name_v1_3[MAX_STRING], name_v2_3[MAX_STRING], str_3[2 * MAX_STRING + 10000];
    int vid_3;
    double weight_3;
    
    fin_3 = fopen(network_file_3, "rb");
    if (fin_3 == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges_3 = 0;
    while (fgets(str_3, sizeof(str_3), fin_3)) num_edges_3++;
    //fclose(fin_3);
    printf("#edges of network_file_3: %lld          \n", num_edges_3);
    
    edge_source_id_3 = (int *)malloc(num_edges_3*sizeof(int));
    edge_target_id_3 = (int *)malloc(num_edges_3*sizeof(int));
    edge_weight_3 = (double *)malloc(num_edges_3*sizeof(double));
    if (edge_source_id_3 == NULL || edge_target_id_3 == NULL || edge_weight_3 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    fin_3 = fopen(network_file_3, "rb");
    num_vertices_3 = 0;
    for (int k = 0; k != num_edges_3; k++)
    {
        fscanf(fin_3, "%s %s %lf", name_v1_3, name_v2_3, &weight_3);
        
        if (k % 10000 == 0)
        {
            printf("Reading edges of network_file_3: %.3lf%%%c", k / (double)(num_edges_3 + 1) * 100, 13);
            fflush(stdout);
        }
        
        vid_3 = SearchHashTable_3(name_v1_3);
		if (vid_3 == -1) {
			if(name_v1_3[0] == 'l'){
				user_vertices++;
				//printf("user_vertices: %d\n",user_vertices);
			}
			
			vid_3 = AddVertex_3(name_v1_3);
		}
        vertex_3[vid_3].degree += weight_3;
        edge_source_id_3[k] = vid_3;
        
        vid_3 = SearchHashTable_3(name_v2_3);
		if (vid_3 == -1) {
			if(name_v2_3[0] == 'l'){
				user_vertices++;
				//printf("user_vertices: %d\n",user_vertices);
			}
			vid_3 = AddVertex_3(name_v2_3);
		}
        vertex_3[vid_3].degree += weight_3;
        edge_target_id_3[k] = vid_3;
        
        edge_weight_3[k] = weight_3;
    }
    fclose(fin_3);
    printf("#vertices of network_file_3: %d          \n", num_vertices_3);

}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable_1()
{
    alias_1 = (long long *)malloc(num_edges_1*sizeof(long long));
    prob_1 = (double *)malloc(num_edges_1*sizeof(double));
    if (alias_1 == NULL || prob_1 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double *norm_prob = (double*)malloc(num_edges_1*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges_1*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges_1*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    
    for (long long k = 0; k != num_edges_1; k++) sum += edge_weight_1[k];
    for (long long k = 0; k != num_edges_1; k++) norm_prob[k] = edge_weight_1[k] * num_edges_1 / sum;
    
    for (long long k = num_edges_1 - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob_1[cur_small_block] = norm_prob[cur_small_block];
        alias_1[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    
    while (num_large_block) prob_1[large_block[--num_large_block]] = 1;
    while (num_small_block) prob_1[small_block[--num_small_block]] = 1;
    
    free(norm_prob);
    free(small_block);
    free(large_block);
}


/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable_2()
{
    alias_2 = (long long *)malloc(num_edges_2*sizeof(long long));
    prob_2 = (double *)malloc(num_edges_2*sizeof(double));
    if (alias_2 == NULL || prob_2 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double *norm_prob = (double*)malloc(num_edges_2*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges_2*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges_2*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    
    for (long long k = 0; k != num_edges_2; k++) sum += edge_weight_2[k];
    for (long long k = 0; k != num_edges_2; k++) norm_prob[k] = edge_weight_2[k] * num_edges_2 / sum;
    
    for (long long k = num_edges_2 - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob_2[cur_small_block] = norm_prob[cur_small_block];
        alias_2[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    
    while (num_large_block) prob_2[large_block[--num_large_block]] = 1;
    while (num_small_block) prob_2[small_block[--num_small_block]] = 1;
    
    free(norm_prob);
    free(small_block);
    free(large_block);
}


/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable_3()
{
    alias_3 = (long long *)malloc(num_edges_3*sizeof(long long));
    prob_3 = (double *)malloc(num_edges_3*sizeof(double));
    if (alias_3 == NULL || prob_3 == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double *norm_prob = (double*)malloc(num_edges_3*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges_3*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges_3*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    
    for (long long k = 0; k != num_edges_3; k++) sum += edge_weight_3[k];
    for (long long k = 0; k != num_edges_3; k++) norm_prob[k] = edge_weight_3[k] * num_edges_3 / sum;
    
    for (long long k = num_edges_3 - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob_3[cur_small_block] = norm_prob[cur_small_block];
        alias_3[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    
    while (num_large_block) prob_3[large_block[--num_large_block]] = 1;
    while (num_small_block) prob_3[small_block[--num_small_block]] = 1;
    
    free(norm_prob);
    free(small_block);
    free(large_block);
}


long long SampleAnEdge_1(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges_1 * rand_value1;
    return rand_value2 < prob_1[k] ? k : alias_1[k];
}

long long SampleAnEdge_2(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges_2 * rand_value1;
    return rand_value2 < prob_2[k] ? k : alias_2[k];
}

long long SampleAnEdge_3(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges_3 * rand_value1;
    return rand_value2 < prob_3[k] ? k : alias_3[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector_1()
{
    long long a, b;
    
    //a = posix_memalign((void **)&emb_vertex_1, 128, (long long)num_vertices_1 * dim * sizeof(real));
	emb_vertex_1 = (real *)_aligned_malloc((long long)num_vertices_1 * dim * sizeof(real), 128);
    if (emb_vertex_1 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (a = 0; a < num_vertices_1; a++) {
		if(vertex_1[a].name[0] == 'l'){
			int index = binarySearch(vertex_1[a].name);
			if(index == -1){
				//printf("the location is %s \n",vertex_1[a].name);
				//printf("Error : cannot find location embedding!\n");
				//system("pause");
				//exit(1);
				for (b = 0; b < dim; b++) 
				emb_vertex_1[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
			}
			else
			for (b = 0; b < dim; b++) 
				emb_vertex_1[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_vertex_1[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}
    
    //a = posix_memalign((void **)&emb_context_1, 128, (long long)num_vertices_1 * dim * sizeof(real));
	emb_context_1 = (real *)_aligned_malloc((long long)num_vertices_1 * dim * sizeof(real), 128);
    if (emb_context_1 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (a = 0; a < num_vertices_1; a++) {
		if(vertex_1[a].name[0] == 'l'){
			int index = binarySearch(vertex_1[a].name);
			if(index == -1){
				for (b = 0; b < dim; b++) 
					emb_context_1[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
				//printf("Error : cannot find location embedding!\n");
				//exit(1);
			}
			else
			for (b = 0; b < dim; b++) 
				emb_context_1[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_context_1[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
    //emb_context[a * dim + b] = 0; //why emb_contexts is set to 0? not emb_vertex?
	}
}

/* Initialize the vertex embedding and the context embedding */
void InitVector_2(){
    long long a, b;
    //a = posix_memalign((void **)&emb_vertex_2, 128, (long long)num_vertices_2 * dim * sizeof(real));
	emb_vertex_2 = (real *)_aligned_malloc((long long)num_vertices_2 * dim * sizeof(real), 128);
    if (emb_vertex_2 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (a = 0; a < num_vertices_2; a++) {
		if(vertex_2[a].name[0] == 'l'){
			int index = binarySearch(vertex_2[a].name);
			if(index == -1){
				for (b = 0; b < dim; b++) 
					emb_vertex_2[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
				//printf("Error : cannot find location embedding!\n");
				//exit(1);
			}
			else
			for (b = 0; b < dim; b++) 
				emb_vertex_2[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_vertex_2[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}
    
    //a = posix_memalign((void **)&emb_context_2, 128, (long long)num_vertices_2 * dim * sizeof(real));
	emb_context_2 = (real *)_aligned_malloc((long long)num_vertices_2 * dim * sizeof(real), 128);
    if (emb_context_2 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (a = 0; a < num_vertices_2; a++) {
		if(vertex_2[a].name[0] == 'l'){
			int index = binarySearch(vertex_2[a].name);
			if(index == -1){
				for (b = 0; b < dim; b++) 
					emb_context_2[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
				//printf("Error : cannot find location embedding!\n");
				//exit(1);
			}
			else
			for (b = 0; b < dim; b++) 
				emb_context_2[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_context_2[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}
    
}


/* Initialize the vertex embedding and the context embedding */
void InitVector_3(){
    long long a, b;
    //a = posix_memalign((void **)&emb_vertex_3, 128, (long long)num_vertices_3 * dim * sizeof(real));
	emb_vertex_3 = (real *)_aligned_malloc((long long)num_vertices_3 * dim * sizeof(real), 128);
    if (emb_vertex_3 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }

    for (a = 0; a < num_vertices_3; a++) {
		if(vertex_3[a].name[0] == 'l'){
			int index = binarySearch(vertex_3[a].name);
			if(index == -1){
				for (b = 0; b < dim; b++) 
					emb_vertex_3[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
				//printf("Error : cannot find location embedding!\n");
				//exit(1);
			}
			else
			for (b = 0; b < dim; b++) 
				emb_vertex_3[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_vertex_3[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}
    
    //a = posix_memalign((void **)&emb_context_3, 128, (long long)num_vertices_3 * dim * sizeof(real));
	emb_context_3 = (real *)_aligned_malloc((long long)num_vertices_3 * dim * sizeof(real), 128);
    if (emb_context_3 == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	
    for (a = 0; a < num_vertices_3; a++) {
		if(vertex_3[a].name[0] == 'l'){
			int index = binarySearch(vertex_3[a].name);
			if(index == -1){
				//printf("Error : cannot find location embedding!\n");
				for (b = 0; b < dim; b++) 
					emb_context_3[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
				//exit(1);
			}
			else
			for (b = 0; b < dim; b++) 
				emb_context_3[a * dim + b] = emb_location[index * dim + b];
		}
		else{
			for (b = 0; b < dim; b++) 
				emb_context_3[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}

}

void InitVector_location(){
	char name[MAX_STRING], ch;
    long long a, b;
	real vec = 0;
    //a = posix_memalign((void **)&emb_vertex_3, 128, (long long)num_vertices_3 * dim * sizeof(real));
	emb_location = (real *)_aligned_malloc((long long)num_location * dim * sizeof(real), 128);
    if (emb_location == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    
	FILE *fp = fopen(embedding_location_file,"rb");

	for (a = 0; a < num_location; a++) {		
		fscanf(fp, "%s%c", name, &ch);
		strcpy(location_name[a] , name);
		for (b = 0; b < dim; b++) {
			fscanf(fp, "%f%c",&vec,&ch);
			emb_location[a * dim + b] = vec;
		}
	}
}


/* Sample negative vertex samples according to vertex degrees */
void InitNegTable_1()
{
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table_1 = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices_1; k++) sum += pow(vertex_1[k].degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(vertex_1[vid].degree, NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        neg_table_1[k] = vid - 1;
    }
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable_2()
{
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table_2 = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices_2; k++) sum += pow(vertex_2[k].degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(vertex_2[vid].degree, NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        neg_table_2[k] = vid - 1;
    }
}


/* Sample negative vertex samples according to vertex degrees */
void InitNegTable_3()
{
    double sum = 0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table_3 = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != num_vertices_3; k++) sum += pow(vertex_3[k].degree, NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(vertex_3[vid].degree, NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid++;
        }
        neg_table_3[k] = vid - 1;
    }
}


/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
    real x = 0, g;
    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    g = (label - FastSigmoid(x)) * rho;
    for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
    for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(void *id)
{
    long long u_1, u_2, u_3, v_1, v_2, v_3, v_1_2, v_1_3, v_2_1, v_2_3, v_3_1, v_3_2, lu, lv=-1, lv_1_2=-1, lv_1_3=-1, lv_3_1=-1, lv_3_2=-1, lv_2_1=-1, lv_2_3=-1, target;
    int label;
    long long count = 0, last_count = 0, curedge_1, curedge_2, curedge_3;
    unsigned long long seed = (long long)id;
    real *vec_error = (real *)calloc(dim, sizeof(real));
    int cc = 0;
    int tt = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    
    while (1)
    {
        tt++;
        //judge for exit
        if (count > total_samples / num_threads + 2) break;
        
        if (count - last_count>10000)
        {
            current_sample_count += count - last_count;
            last_count = count;
            printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
            fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }
        
        /*
         * the first graph should be sampled at each iteration
         */
        
		curedge_1 = SampleAnEdge_1(uni(), uni());
        u_1 = edge_source_id_1[curedge_1];
        v_1 = edge_target_id_1[curedge_1];
        v_1_2 = SearchHashTable_2(vertex_1[v_1].name);
        v_1_3 = SearchHashTable_3(vertex_1[v_1].name);
        if(v_1_2 != -1) lv_1_2 = v_1_2*dim;
        if(v_1_3 != -1) lv_1_3 = v_1_3*dim;
        
        //char* t2 = vertex_1[v_1].name;
        //char* t3 = vertex_2[v_1_2].name;
        //char* t4 = vertex_3[v_1_3].name;
        
        lu = u_1 * dim;
        for (int c = 0; c != dim; c++) vec_error[c] = 0;
        
        // NEGATIVE SAMPLING
        for (int d = 0; d != num_negative + 1; d++)
        {
            if (d == 0)
            {
                target = v_1;
                label = 1;
            }
            else
            {
                target = neg_table_1[Rand(seed)];
                label = 0;
            }
            lv = target * dim;
            //if (order == 1) Update(&emb_vertex_1[lu], &emb_vertex_1[lv], vec_error, label);
            if (order == 2) Update(&emb_vertex_1[lu], &emb_context_1[lv], vec_error, label);
        }
        for (int c = 0; c != dim; c++) emb_vertex_1[c + lu] += vec_error[c];
        
        if(v_1_2 != -1) emb_context_2[lv_1_2] = emb_context_1[lv];
        if(v_1_3 != -1) emb_context_3[lv_1_3] = emb_context_1[lv];
        
        
        //*******************
        /*
         * the second graph can be sampled based on a paratmeter lambda_1, which decide the proablity of be sampled.
         */
        
        float lambda_1=0.01;
        if(dis(gen)<=lambda_1){
            //printf("%f\n",b);
            cc++;
            curedge_2 = SampleAnEdge_2(uni(), uni());
            u_2 = edge_source_id_2[curedge_2];
            v_2 = edge_target_id_2[curedge_2];
            v_2_1 = SearchHashTable_1(vertex_2[v_2].name);
            v_2_3 = SearchHashTable_3(vertex_2[v_2].name);
            if(v_2_1 != -1) lv_2_1 = v_2_1*dim;
            if(v_2_3 != -1) lv_2_3 = v_2_3*dim;
        
            lu = u_2 * dim;
            for (int c = 0; c != dim; c++) vec_error[c] = 0;
        
            // NEGATIVE SAMPLING
            for (int d = 0; d != num_negative + 1; d++)
            {
                if (d == 0)
                {
                    target = v_2;
                    label = 1;
                }
                else
                {
                    target = neg_table_2[Rand(seed)];
                    label = 0;
                }
                lv = target * dim;
                //if (order == 1) Update(&emb_vertex_1[lu], &emb_vertex_1[lv], vec_error, label);
                if (order == 2) Update(&emb_vertex_2[lu], &emb_context_2[lv], vec_error, label);
            }
            for (int c = 0; c != dim; c++) emb_vertex_2[c + lu] += vec_error[c];
        
            if(v_2_1 != -1) emb_context_1[lv_2_1] = emb_context_2[lv];
            if(v_2_3 != -1) emb_context_3[lv_2_3] = emb_context_2[lv];
        }
        
        //*********************************
        /*
         * the second graph can be sampled based on a paratmeter lambda_2, which decide the proablity of be sampled.
         */
        float lambda_2=0.001;
        if(dis(gen)<=lambda_2){
            curedge_3 = SampleAnEdge_3(uni(), uni());
            u_3 = edge_source_id_3[curedge_3];
            v_3 = edge_target_id_3[curedge_3];
            v_3_1 = SearchHashTable_1(vertex_3[v_3].name);
            v_3_2 = SearchHashTable_2(vertex_3[v_3].name);
            if(v_3_1 != -1) lv_3_1 = v_3_1*dim;
            if(v_3_2 != -1) lv_3_2 = v_3_2*dim;
        
            lu = u_3 * dim;
            for (int c = 0; c != dim; c++) vec_error[c] = 0;
        
            // NEGATIVE SAMPLING
            for (int d = 0; d != num_negative + 1; d++)
            {
                if (d == 0)
                {
                    target = v_3;
                    label = 1;
                }
                else
                {
                    target = neg_table_3[Rand(seed)];
                    label = 0;
                }
                lv = target * dim;
                //if (order == 1) Update(&emb_vertex_1[lu], &emb_vertex_1[lv], vec_error, label);
                if (order == 2) Update(&emb_vertex_3[lu], &emb_context_3[lv], vec_error, label);
            }
            for (int c = 0; c != dim; c++) emb_vertex_3[c + lu] += vec_error[c];
        
            if(lv_3_1 != -1) emb_context_1[lv_3_1] = emb_context_3[lv];
            if(lv_3_2 != -1) emb_context_2[lv_3_2] = emb_context_3[lv];
        }
        
        count++;
    }
    //printf("%d\n",cc);
    //printf("****");
    //printf("%d\n",tt);
    free(vec_error);
    return NULL;
}

void Output()
{
  
	printf("step one\n");
	//sequenceVertex = (struct SequenceVertex *)calloc(max_num_vertices, sizeof(struct SequenceVertex));
	//context 100
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", user_vertices, dim);
	for (int a = 0; a < num_vertices_3; a++)
	{
		if(vertex_3[a].name[0] == 'l'){
			fprintf(fo, "%s ", vertex_3[a].name);
			if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_context_3[a * dim + b], sizeof(real), 1, fo);
			else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_context_3[a * dim + b]);
			fprintf(fo, "\n");
		}
	}
	fclose(fo);
	//printf("step two\n");
	//context 100 for binary
	/*
	fo = fopen("loc_time_0.00001_binary.txt", "wb");
	fprintf(fo, "%d %d\n", user_vertices, dim);
	for (int a = 0; a < num_vertices_3; a++)
	{
		if(vertex_3[a].name[0] == 'l'){
			fprintf(fo, "%s ", vertex_3[a].name);
			if (1) for (int b = 0; b < dim; b++) fwrite(&emb_context_3[a * dim + b], sizeof(real), 1, fo);
			else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_context_3[a * dim + b]);
			fprintf(fo, "\n");
		}
	}
    fclose(fo);
	*/
}

void Output_test()
{
	printf("step zero\n");
	//sequenceVertex = (struct SequenceVertex *)calloc(max_num_vertices, sizeof(struct SequenceVertex));
	//context 100
	FILE *fo = fopen("vertex_name.txt", "wb");
	//fprintf(fo, "%d %d\n", user_vertices, dim);
	for (int a = 0; a < num_vertices_3; a++)
	{	
		fprintf(fo, "%s ", vertex_3[a].name);
		for(int b = 0 ; b < 200 ; b++){
			fprintf(fo, "%f", emb_context_3[a * 200 + b]);
			if(b == 199)
				fprintf(fo, "\n");
			else
				fprintf(fo, " ");
		}
	}
	fclose(fo);
	exit(3);
}

void TrainLINE() {
    long a;
	boost::thread *pt = new boost::thread[num_threads];
    //pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    if (order != 2)
    {
        printf("Error: order should be 2!\n"); //either or
        exit(1);
    }
    printf("--------------------------------\n");
    printf("Order: %d\n", order);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("--------------------------------\n");
    
    InitHashTable_1(); //create a HashTable for recording nodes' names
    InitHashTable_2();
    InitHashTable_3();
    ReadData(); //read in vertices and edges of all networks
    InitAliasTable_1();
    InitAliasTable_2();
    InitAliasTable_3();

    InitVector_location();
	
    InitVector_1();
    InitVector_2();
    InitVector_3();
	
	printf("Have been initialed!\n");

    InitNegTable_1();
    InitNegTable_2();
    InitNegTable_3();
    
    InitSigmoidTable();
    
    //gsl_rng_env_setup();
    //gsl_T = gsl_rng_rand48;
    //gsl_r = gsl_rng_alloc(gsl_T);
    //gsl_rng_set(gsl_r, 314159265);
    clock_t start = clock();
    printf("--------------------------------\n");
    for (a = 0; a < num_threads; a++) pt[a] = boost::thread(TrainLINEThread, (void *)a);//pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pt[a].join();//pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    //Output_test();
    Output();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("LINETHREE: Dealing three bipartite graph Embedding\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train1 <file>\n");
        printf("\t\tUse firt network data from <file> to train the model\n");
        printf("\t-train2 <file>\n");
        printf("\t\tUse second network data from <file> to train the model\n");
        printf("\t-train3 <file>\n");
        printf("\t\tUse third network data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-order <int>\n");
        printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-rho <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
        
        //dim = 200;
        //real x = RAND_MAX;//(rand() / (real)RAND_MAX - 0.5) / dim;
        //printf("%f",x);
        //return 0;
    }
    
    if ((i = ArgPos((char *)"-train1", argc, argv)) > 0) strcpy(network_file_1, argv[i + 1]);
    if ((i = ArgPos((char *)"-train2", argc, argv)) > 0) strcpy(network_file_2, argv[i + 1]);
    if ((i = ArgPos((char *)"-train3", argc, argv)) > 0) strcpy(network_file_3, argv[i + 1]);
	if ((i = ArgPos((char *)"-intial", argc, argv)) > 0) strcpy(embedding_location_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
   
    total_samples = (total_samples)*1000000; // the base of the number of the samples
    rho = init_rho;
    
    vertex_1= (struct ClassVertex *)calloc(max_num_vertices_1, sizeof(struct ClassVertex)); //one of different network
    vertex_2= (struct ClassVertex *)calloc(max_num_vertices_2, sizeof(struct ClassVertex));
    vertex_3= (struct ClassVertex *)calloc(max_num_vertices_3, sizeof(struct ClassVertex));
    
    TrainLINE();
	
    return 0;
}
