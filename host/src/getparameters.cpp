/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
Copyright (C) 2022 Intel Corporation

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/
#ifdef __INTEL_LLVM_COMPILER
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#endif
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <locale>

#include "getparameters.h"
#ifdef __INTEL_LLVM_COMPILER
#include <cmath>
#endif

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

int dpf_token(const char* token)
{
	const struct {
		const char* string;
		const int   token;
		bool        evaluate; // need to evaluate parameters (and either use it or make sure it's a certain value)
	} supported_dpf_tokens [] = {
	      {"ligand",             DPF_MOVE,              true},  /* movable ligand file name */
	      {"move",               DPF_MOVE,              true},  /* movable ligand file name */
	      {"fld",                DPF_FLD,               true},  /* grid data file name */
	      {"map",                DPF_MAP,               true},  /* grid map specifier */
	      {"about",              DPF_ABOUT,             false}, /* rotate about */
	      {"tran0",              DPF_TRAN0,             true},  /* translate (needs to be "random") */
	      {"axisangle0",         DPF_AXISANGLE0,        true},  /* rotation axisangle (needs to be "random") */
	      {"quaternion0",        DPF_QUATERNION0,       true},  /* quaternion (of rotation, needs to be "random") */
	      {"quat0",              DPF_QUAT0,             true},  /* quaternion (of rotation, needs to be "random") */
	      {"dihe0",              DPF_DIHE0,             true},  /* number of dihedrals (needs to be "random") */
	      {"ndihe",              DPF_NDIHE,             false}, /* number of dihedrals (is in pdbqt) */
	      {"torsdof",            DPF_TORSDOF,           false}, /* torsional degrees of freedom (is in pdbqt) */
	      {"intnbp_coeffs",      DPF_INTNBP_COEFFS,     true},  /* internal pair energy coefficients */
	      {"intnbp_r_eps",       DPF_INTNBP_REQM_EPS,   true},  /* internal pair energy coefficients */
	      {"runs",               DPF_RUNS,              true},  /* number of runs */
	      {"ga_run",             DPF_GALS,              true},  /* run a number of runs */
	      {"gals_run",           DPF_GALS,              true},  /* run a number of runs */
	      {"outlev",             DPF_OUTLEV,            false}, /* output level */
	      {"rmstol",             DPF_RMSTOL,            true},  /* RMSD cluster tolerance */
	      {"extnrg",             DPF_EXTNRG,            false}, /* external grid energy */
	      {"intelec",            DPF_INTELEC,           true},  /* calculate ES energy (needs not be "off") */
	      {"smooth",             DPF_SMOOTH,            true},  /* smoothing range */
	      {"seed",               DPF_SEED,              true},  /* random number seed */
	      {"e0max",              DPF_E0MAX,             false}, /* simanneal max inital energy (ignored) */
	      {"set_ga",             DPF_SET_GA,            false}, /* use genetic algorithm (yes, that's us) */
	      {"set_sw1",            DPF_SET_SW1,           false}, /* use Solis-Wets (we are by default)*/
	      {"set_psw1",           DPF_SET_PSW1,          false}, /* use pseudo Solis-Wets (nope, SW) */
	      {"analysis",           DPF_ANALYSIS,          false}, /* analysis data (we're doing it) */
	      {"ga_pop_size",        GA_pop_size,           true},  /* population size */
	      {"ga_num_generations", GA_num_generations,    true},  /* number of generations */
	      {"ga_num_evals",       GA_num_evals,          true},  /* number of evals */
	      {"ga_window_size",     GA_window_size,        false}, /* genetic algorithm window size */
	      {"ga_elitism",         GA_elitism,            false}, /* GA parameters: */
	      {"ga_mutation_rate",   GA_mutation_rate,      true},  /*     The ones set to true */
	      {"ga_crossover_rate",  GA_crossover_rate,     true},  /*     have a corresponding */
	      {"ga_cauchy_alpha",    GA_Cauchy_alpha,       false}, /*     parameter in AD-GPU  */
	      {"ga_cauchy_beta",     GA_Cauchy_beta,        false}, /*     the others ignored   */
	      {"sw_max_its",         SW_max_its,            true},  /* local search iterations */
	      {"sw_max_succ",        SW_max_succ,           true},  /* cons. success limit */
	      {"sw_max_fail",        SW_max_fail,           true},  /* cons. failure limit */
	      {"sw_rho",             SW_rho,                false}, /* rho - is 1.0 here */
	      {"sw_lb_rho",          SW_lb_rho,             true},  /* lower bound of rho */
	      {"ls_search_freq",     LS_search_freq,        false}, /* ignored as likely wrong for algorithm here */
	      {"parameter_file",     DPF_PARAMETER_LIBRARY, false}, /* parameter file (use internal currently) */
	      {"ligand_types",       DPF_LIGAND_TYPES,      true},  /* ligand types used */
	      {"output_pop_file",    DPF_POPFILE,           false}, /* output population to file */
	      {"flexible_residues",  DPF_FLEXRES,           true},  /* flexibe residue file name */
	      {"flexres",            DPF_FLEXRES,           true},  /* flexibe residue file name */
	      {"elecmap",            DPF_ELECMAP,           false}, /* electrostatic grid map (we use fld file basename) */
	      {"desolvmap",          DPF_DESOLVMAP,         false}, /* desolvation grid map (we use fld file basename) */
	      {"dsolvmap",           DPF_DESOLVMAP,         false}, /* desolvation grid map (we use fld file basename) */
	      {"unbound_model",      DPF_UNBOUND_MODEL,     true}   /* unbound model (bound|extended|compact) */
	                            };

	if (token[0]=='\0')
		return DPF_BLANK_LINE;
	if (token[0]=='#')
		return DPF_COMMENT;

	for (int i=0; i<(int)(sizeof(supported_dpf_tokens)/sizeof(*supported_dpf_tokens)); i++){
		if(stricmp(supported_dpf_tokens[i].string,token) == 0){
			if(supported_dpf_tokens[i].evaluate)
				return supported_dpf_tokens[i].token;
			else
				return DPF_NULL;
			break; // found
		}
	}

	return DPF_UNKNOWN;
}

int preparse_dpf(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                )
// This function checks if a dpf file is used and, if runs are specified, map and ligand information
// is stored in the filelist; flexres information and which location in the dpf parameters are in each
// run is stored separately to allow logical parsing with the correct parameters initialized per run
{
	bool output_multiple_warning = true;
	for (int i=1; i<(*argc)-1; i++)
	{
		// Argument: dpf file name.
		if (strcmp("-import_dpf", argv[i]) == 0){
			if(mypars->dpffile){
				free(mypars->dpffile);
				if(output_multiple_warning){
					printf("Warning: Multiple -import_dpf arguments, only the last one will be used.");
					output_multiple_warning = false;
				}
			}
			mypars->dpffile = strdup(argv[i+1]);
		}
	}
	if (mypars->dpffile){
		std::ifstream file(mypars->dpffile);
		if(file.fail()){
			printf("\nError: Could not open dpf file %s. Check path and permissions.\n",mypars->dpffile);
			return 1;
		}
		mypars->elec_min_distance = 0.5; // default for AD4
		std::string line;
		char tempstr[256], argstr[256];
		char* args[2];
		int tempint, i, len;
		float tempfloat;
		int line_count = 0;
		int ltype_nr = 0;
		int mtype_nr = 0;
		char ltypes[MAX_NUM_OF_ATYPES][4];
		char* typestr;
		memset(ltypes,0,4*MAX_NUM_OF_ATYPES*sizeof(char));
		unsigned int idx;
		pair_mod* curr_pair;
		float paramA, paramB;
		int m, n;
		char typeA[4], typeB[4];
		filelist.max_len = 256;
		bool new_device = false; // indicate if current mypars has a new device requested
		while(std::getline(file, line)) {
			line_count++;
			trim(line); // Remove leading and trailing whitespace
			tempstr[0]='\0';
			sscanf(line.c_str(),"%255s",tempstr);
			int token_id = dpf_token(tempstr);
			switch(token_id){
				case DPF_MOVE: // movable ligand file name
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(mypars->ligandfile) free(mypars->ligandfile);
						mypars->ligandfile = strdup(argstr);
						break;
				case DPF_FLEXRES: // flexibe residue file name
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(mypars->flexresfile) free(mypars->flexresfile);
						mypars->flexresfile = strdup(argstr);
						break;
				case DPF_FLD: // grid data file name
						sscanf(line.c_str(),"%*s %255s",argstr);
						// Add the .fld file
						if(mypars->fldfile) free(mypars->fldfile);
						mypars->fldfile = strdup(argstr); // this allows using the dpf to set up all parameters but the ligand
						// Filling mygrid according to the specified fld file
						mygrid->info_read = false;
						if (get_gridinfo(mypars->fldfile, mygrid) != 0)
						{
							printf("\nError: get_gridinfo failed with fld file specified with <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_LIGAND_TYPES: // ligand types used
						len=-1;
						for(i=strlen(tempstr); i<line.size(); i++){
							if(isspace(line[i])){ // whitespace
								len=-1;
							} else{ // not whitespace aka an atom type
								if(len<0){ // new type starts
									len=i;
									ltype_nr++;
								}
								if(i-len<3){
									ltypes[ltype_nr-1][i-len] = line[i];
								} else{
									printf("\nError: Atom types are limited to 3 characters in <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
									return 1;
								}
							}
						}
						break;
				case DPF_MAP: // grid map specifier
						sscanf(line.c_str(),"%*s %255s",argstr);
						argstr[strlen(argstr)-4] = '\0'; // get rid of .map extension
						typestr=strchr(argstr+strlen(argstr)-4,'.')+1; // 4 chars for atom type
						if(mtype_nr>=ltype_nr){
							printf("\nError: More map files specified than atom types at %s:%u (ligand types need to be specified before maps).\n",mypars->dpffile,line_count);
							return 1;
						}
						if(strcmp(typestr,ltypes[mtype_nr])){ // derived type
							if(mypars->nr_deriv_atypes==0){ // get the derived atom types started
								mypars->deriv_atypes=(deriv_atype*)malloc(sizeof(deriv_atype));
								if(mypars->deriv_atypes==NULL){
									printf("Error: Cannot allocate memory for derivative type.\n");
									return 1;
								}
							}
							if(!add_deriv_atype(mypars,ltypes[mtype_nr],strlen(ltypes[mtype_nr]))){
								printf("Error: Derivative (ligand type %s) names can only be upto 3 characters long.\n",ltypes[mtype_nr]);
								return 1;
							}
							idx = mypars->nr_deriv_atypes-1;
							strcpy(mypars->deriv_atypes[idx].base_name,typestr);
#ifdef DERIVTYPE_INFO
							printf("%i: %s=%s\n",mypars->deriv_atypes[idx].nr,mypars->deriv_atypes[idx].deriv_name,mypars->deriv_atypes[idx].base_name);
#endif
						}
						mtype_nr++;
						break;
				case DPF_INTNBP_COEFFS: // internal pair energy coefficients
				case DPF_INTNBP_REQM_EPS: // internal pair energy coefficients
						if(sscanf(line.c_str(), "%*s %f %f %d %d %3s %3s", &paramA, &paramB, &m, &n, typeA, typeB)<6){
							printf("Error: Syntax error for <%s>, 6 values are required at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						if(m==n){
							printf("Error: Syntax error for <%s>, exponents need to be different at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						if(token_id==DPF_INTNBP_COEFFS){
							tempfloat = pow(paramB/paramA*n/m,m-n); // reqm
							paramA = paramB*float(m-n)/(pow(tempfloat,n)*m); // epsAB
							paramB = tempfloat; // rAB
						}
						// parameters are sorted out, now add to modpairs
						mypars->nr_mod_atype_pairs++;
						if(mypars->nr_mod_atype_pairs==1)
							mypars->mod_atype_pairs=(pair_mod*)malloc(sizeof(pair_mod));
						else
							mypars->mod_atype_pairs=(pair_mod*)realloc(mypars->mod_atype_pairs, mypars->nr_mod_atype_pairs*sizeof(pair_mod));
						if(mypars->mod_atype_pairs==NULL){
							printf("Error: Cannot allocate memory for <%s> pair energy modification.\n at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						curr_pair=&mypars->mod_atype_pairs[mypars->nr_mod_atype_pairs-1];
						strcpy(curr_pair->A,typeA);
						strcpy(curr_pair->B,typeB);
						curr_pair->nr_parameters=4;
						curr_pair->parameters=(float*)malloc(curr_pair->nr_parameters*sizeof(float));
						if(curr_pair->parameters==NULL){
							printf("Error: Cannot allocate memory for <%s> pair energy modification.\n at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						curr_pair->parameters[0]=paramA;
						curr_pair->parameters[1]=paramB;
						curr_pair->parameters[2]=m;
						curr_pair->parameters[3]=n;
#ifdef MODPAIR_INFO
						printf("%i: %s:%s",mypars->nr_mod_atype_pairs,curr_pair->A,curr_pair->B);
						for(idx=0; idx<curr_pair->nr_parameters; idx++)
							printf(",%f",curr_pair->parameters[idx]);
						printf("\n");
#endif
						break;
				case DPF_TRAN0: // translate                     (needs to be "random")
				case DPF_AXISANGLE0: // rotation axisangle       (needs to be "random")
				case DPF_QUATERNION0: // quaternion (of rotation, needs to be "random")
				case DPF_QUAT0: // quaternion       (of rotation, needs to be "random")
				case DPF_DIHE0: // number of dihedrals           (needs to be "random")
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"random")){
							printf("\nError: Currently only \"random\" is supported as <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_RUNS: // set number of runs
				case DPF_GALS: // actually run a search
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS))
							mypars->num_of_runs = (int) tempint;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be an integer between 1 and %d.\n",tempstr,mypars->dpffile,line_count,MAX_NUM_OF_RUNS);
						if(token_id!=DPF_RUNS){
							// Add the fld file to use
							if (!mypars->fldfile){
								printf("\nError: No map file on record yet. Please specify a map file before the first ligand.\n");
								return 1;
							}
							filelist.fld_files.push_back(mypars->fldfile);
							// If more than one unique protein, cant do map preloading yet
							if (filelist.fld_files.size()>1){
								filelist.preload_maps=false;
							}
							// Add the ligand to filelist
							filelist.ligand_files.push_back(mypars->ligandfile);
							// Default resname is filelist basename
							if(mypars->resname) free(mypars->resname);
							len=strlen(mypars->ligandfile)-6; // .pdbqt = 6 chars
							if(len>0){
								mypars->resname = (char*)malloc((len+1)*sizeof(char));
								strncpy(mypars->resname,mypars->ligandfile,len); // Default is ligand file basename
								mypars->resname[len]='\0';
							} else mypars->resname = strdup("docking"); // Fallback to old default
							filelist.resnames.push_back(mypars->resname);
							if(new_device) mypars->devices_requested++;
							// Before pushing parameters and grids back make sure
							// the filename pointers are unique
							if(filelist.mypars.size()>0){ // mypars and mygrids have same size
								if((filelist.mypars.back().flexresfile) &&
								   (filelist.mypars.back().flexresfile==mypars->flexresfile))
									mypars->flexresfile=strdup(mypars->flexresfile);
								if((filelist.mypars.back().xrayligandfile) &&
								   (filelist.mypars.back().xrayligandfile==mypars->xrayligandfile))
									mypars->xrayligandfile=strdup(mypars->xrayligandfile);
								if((filelist.mygrids.back().grid_file_path) &&
								   (filelist.mygrids.back().grid_file_path==mygrid->grid_file_path))
									mygrid->grid_file_path=strdup(mygrid->grid_file_path);
								if((filelist.mygrids.back().receptor_name) &&
								   (filelist.mygrids.back().receptor_name==mygrid->receptor_name))
									mygrid->receptor_name=strdup(mygrid->receptor_name);
								if((filelist.mygrids.back().map_base_name) &&
								   (filelist.mygrids.back().map_base_name==mygrid->map_base_name))
									mygrid->map_base_name=strdup(mygrid->map_base_name);
							}
							// Add the parameter block now that resname is set
							filelist.mypars.push_back(*mypars);
							// Also add the grid
							filelist.mygrids.push_back(*mygrid);
						}
						break;
				case DPF_INTELEC: // calculate ES energy (needs not be "off")
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"off")==0){
							printf("\nError: \"Off\" is not supported as <%s> parameter at %s:%u.\n",tempstr,mypars->dpffile,line_count);
							return 1;
						}
						break;
				case DPF_SMOOTH: // smoothing range
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						// smooth is measured in Angstrom
						if ((tempfloat >= 0.0f) && (tempfloat <= 0.5f))
							mypars->smooth = tempfloat;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be a float between 0 and 0.5.\n",tempstr,mypars->dpffile,line_count);
						break;
				case DPF_SEED: // random number seed
						m=0; n=0; i=0;
						if(sscanf(line.c_str(),"%*s %d %d %d",&m, &n, &i)>0){ // one or more numbers
							mypars->seed[0]=m; mypars->seed[1]=n; mypars->seed[2]=i;
						} else
							printf("Warning: Only numerical values currently supported for <%s> at %s:%u.\n",tempstr,mypars->dpffile,line_count);
						break;
				case DPF_RMSTOL: // RMSD clustering tolerance
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						if (tempfloat > 0.0)
							mypars->rmsd_tolerance = tempfloat;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be greater than 0.\n",tempstr,mypars->dpffile,line_count);
						break;
				case GA_pop_size: // population size
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint >= 2) && (tempint <= MAX_POPSIZE))
							mypars->pop_size = (unsigned long) (tempint);
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be an integer between 2 and %d.\n",tempstr,mypars->dpffile,line_count,MAX_POPSIZE);
						break;
				case GA_num_generations: // number of generations
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 16250000))
							mypars->num_of_generations = (unsigned long) tempint;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be between 0 and 16250000.\n",tempstr,mypars->dpffile,line_count);
						break;
				case GA_num_evals: // number of evals
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 0x7FFFFFFF)){
							mypars->num_of_energy_evals = (unsigned long) tempint;
							mypars->nev_provided = true;
						} else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be between 0 and 2^31-1.\n",tempstr,mypars->dpffile,line_count);
						break;
				case GA_mutation_rate: // mutation rate
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						tempfloat*=100.0;
						if ((tempfloat >= 0.0) && (tempfloat < 100.0))
							mypars->mutation_rate = tempfloat;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
						break;
				case GA_crossover_rate: // crossover rate
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						tempfloat*=100.0;
						if ((tempfloat >= 0.0) && (tempfloat <= 100.0))
							mypars->crossover_rate = tempfloat;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
						break;
				case SW_max_its: // local search iterations
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 262144))
							mypars->max_num_of_iters = (unsigned long) tempint;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be an integer between 1 and 262143.\n",tempstr,mypars->dpffile,line_count);
						break;
				case SW_max_succ: // cons. success limit
				case SW_max_fail: // cons. failure limit
						sscanf(line.c_str(),"%*s %d",&tempint);
						if ((tempint > 0) && (tempint < 256))
							mypars->cons_limit = (unsigned long) (tempint);
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be an integer between 1 and 255.\n",tempstr,mypars->dpffile,line_count);
						break;
				case SW_lb_rho: // lower bound of rho
						sscanf(line.c_str(),"%*s %f",&tempfloat);
						if ((tempfloat >= 0.0) && (tempfloat < 1.0))
							mypars->rho_lower_bound = tempfloat;
						else
							printf("Warning: value of <%s> at %s:%u ignored. Value must be a float between 0 and 1.\n",tempstr,mypars->dpffile,line_count);
						break;
				case DPF_UNBOUND_MODEL: // unbound model (bound|extended|compact)
						sscanf(line.c_str(),"%*s %255s",argstr);
						if(stricmp(argstr,"bound")==0){
							mypars->unbound_model = 0;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else if(stricmp(argstr,"extended")==0){
							mypars->unbound_model = 1;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else if(stricmp(argstr,"compact")==0){
							mypars->unbound_model = 2;
							mypars->coeffs = unbound_models[mypars->unbound_model];
						} else{
							printf("Error: Unsupported value for <%s> at %s:%u. Value must be one of (bound|extend|compact).\n",tempstr,mypars->dpffile,line_count);
						}
						break;
				case DPF_COMMENT: // we use comments to allow specifying AD-GPU command lines
						sscanf(line.c_str(),"%*s %255s %255s",tempstr,argstr);
						if(tempstr[0]=='-'){ // potential command line argument
							i=2; // one command line argument to be parsed
							args[0]=tempstr;
							args[1]=argstr;
							if(get_commandpars(&i,args,&(mygrid->spacing),mypars,false)<2){
								printf("Warning: Command line option '%s' at %s:%u is not supported inside a dpf file.\n",tempstr,mypars->dpffile,line_count);
							}
							// count GPUs in case we set a different one
							if(strcmp(tempstr,"-devnum")==0){
								new_device=false;
								for(i=0; (i<filelist.mypars.size())&&!new_device; i++){
									if(mypars->devnum==filelist.mypars[i].devnum){
										new_device=true;
									}
								}
							}
						}
						break;
				case DPF_UNKNOWN: // error condition
						printf("\nError: Unknown or unsupported dpf token <%s> at %s:%u.\n",tempstr,mypars->dpffile,line_count);
						return 1;
				default: // means there's a keyword detected that's not yet implemented here
						printf("<%s> has not yet been implemented.\n",tempstr);
				case DPF_BLANK_LINE: // nothing to do here
				case DPF_NULL:
						break;
			}
		}
	}
	filelist.nfiles = filelist.ligand_files.size();
	if(filelist.nfiles>0) filelist.used = true;
	return 0;
}

int get_filelist(
                 const int*      argc,
                       char**    argv,
                       Dockpars* mypars,
                       Gridinfo* mygrid,
                       FileList& filelist
                )
// The function checks if a filelist has been provided according to the proper command line arguments.
// If it is, it loads the .fld, .pdbqt, and resname files into vectors
{
	bool output_multiple_warning = true;
	for (int i=1; i<(*argc)-1; i++)
	{
		// Argument: file name that contains list of files.
		if (strcmp("-filelist", argv[i]) == 0)
		{
			filelist.used = true;
			if(filelist.filename){
				free(filelist.filename);
				if(output_multiple_warning){
					printf("Warning: Multiple -filelist arguments, only the last one will be used.");
					output_multiple_warning = false;
				}
			}
			filelist.filename = strdup(argv[i+1]);
		}
	}

	if (filelist.filename){ // true when -filelist specifies a filename
	                        // filelist.used may be true when dpf file is specified as it uses the filelist to store runs
		std::ifstream file(filelist.filename);
		if(file.fail()){
			printf("\nError: Could not open filelist %s. Check path and permissions.\n",filelist.filename);
			return 1;
		}
		std::string line;
		bool prev_line_was_fld=false;
		unsigned int initial_res_count = filelist.resnames.size();
		int len;
		while(std::getline(file, line)) {
			trim(line); // Remove leading and trailing whitespace
			len = line.size();
			if(len>filelist.max_len) filelist.max_len = len;
			if (len>=4 && line.compare(len-4,4,".fld") == 0){
				if (prev_line_was_fld){ // Overwrite the previous fld file if two in a row
					filelist.fld_files[filelist.fld_files.size()-1] = line;
					printf("\nWarning: a listed .fld file was not used!\n");
				} else {
					// Add the .fld file
					filelist.fld_files.push_back(line);
					prev_line_was_fld=true;

					// If more than one unique protein, cant do map preloading yet
					if (filelist.fld_files.size()>1){
						filelist.preload_maps=false;
					}
				}
				// Filling mygrid according to the specified fld file
				mygrid->info_read = false;
				if (get_gridinfo(filelist.fld_files[filelist.fld_files.size()-1].c_str(), mygrid) != 0)
				{
					printf("\nError: get_gridinfo failed with fld file specified in filelist.\n");
					return 1;
				}
			} else if (len>=6 && line.compare(len-6,6,".pdbqt") == 0){
				// Add the .pdbqt
				filelist.ligand_files.push_back(line);
				// Before pushing parameters and grids back make sure
				// the filename pointers are unique
				if(filelist.mypars.size()>0){ // mypars and mygrids have same size
					if((filelist.mypars.back().flexresfile) &&
					   (filelist.mypars.back().flexresfile==mypars->flexresfile))
						mypars->flexresfile=strdup(mypars->flexresfile);
					if((filelist.mypars.back().xrayligandfile) &&
					   (filelist.mypars.back().xrayligandfile==mypars->xrayligandfile))
						mypars->xrayligandfile=strdup(mypars->xrayligandfile);
					if((filelist.mygrids.back().grid_file_path) &&
					   (filelist.mygrids.back().grid_file_path==mygrid->grid_file_path))
						mygrid->grid_file_path=strdup(mygrid->grid_file_path);
					if((filelist.mygrids.back().receptor_name) &&
					   (filelist.mygrids.back().receptor_name==mygrid->receptor_name))
						mygrid->receptor_name=strdup(mygrid->receptor_name);
					if((filelist.mygrids.back().map_base_name) &&
					   (filelist.mygrids.back().map_base_name==mygrid->map_base_name))
						mygrid->map_base_name=strdup(mygrid->map_base_name);
				}
				// Add the parameter block
				filelist.mypars.push_back(*mypars);
				// Add the grid info
				filelist.mygrids.push_back(*mygrid);
				if (filelist.fld_files.size()==0){
					if(mygrid->info_read){ // already read a map file in with dpf import
						printf("\nUsing map file from dpf import.\n");
						filelist.fld_files.push_back(mypars->fldfile);
					} else{
						printf("\nError: No map file on record yet. Please specify a .fld file before the first ligand (%s).\n",line.c_str());
						return 1;
					}
				}
				if (filelist.ligand_files.size()>filelist.fld_files.size()){
					// If this ligand doesnt have a protein preceding it, use the previous protein
					filelist.fld_files.push_back(filelist.fld_files[filelist.fld_files.size()-1]);
				}
				prev_line_was_fld=false;
			} else if (len>0) {
				// Anything else in the file is assumed to be the resname
				filelist.resnames.push_back(line);
			}
		}

		filelist.nfiles = filelist.ligand_files.size();

		if (filelist.ligand_files.size()==0){
			printf("\nError: No ligands, through lines ending with the .pdbqt suffix, have been specified.\n");
			return 1;
		}
		if (filelist.ligand_files.size() != filelist.resnames.size()){
			if(filelist.resnames.size()-initial_res_count>0){ // make sure correct number of resnames were specified when they were specified
				printf("\nError: Inconsistent number of resnames (%lu) compared to ligands (%lu)!\n",filelist.resnames.size(),filelist.ligand_files.size());
			} else{ // otherwise add default resname (ligand basename)
				for(unsigned int i=filelist.resnames.size(); i<filelist.ligand_files.size(); i++)
					filelist.resnames.push_back(filelist.ligand_files[i].substr(0,filelist.ligand_files[i].size()-6));
			}
			return 1;
		}
		for(unsigned int i=initial_res_count; i<filelist.ligand_files.size(); i++){
			if(filelist.mypars[i].fldfile) free(filelist.mypars[i].fldfile);
			filelist.mypars[i].fldfile = strdup(filelist.fld_files[i].c_str());
			if(filelist.mypars[i].ligandfile) free(filelist.mypars[i].ligandfile);
			filelist.mypars[i].ligandfile = strdup(filelist.ligand_files[i].c_str());
			if(filelist.mypars[i].resname) free(filelist.mypars[i].resname);
			filelist.mypars[i].resname = strdup(filelist.resnames[i].c_str());
		}
	}
	filelist.preload_maps&=filelist.used;

	return 0;
}

int get_filenames_and_ADcoeffs(
                               const int*      argc,
                                     char**    argv,
                                     Dockpars* mypars,
                               const bool      multiple_files
                              )
// The function fills the file name and coeffs fields of mypars parameter
// according to the proper command line arguments.
{
	int i;
	int ffile_given, lfile_given;
	long tempint;

	ffile_given = (mypars->fldfile!=NULL);
	lfile_given = (mypars->ligandfile!=NULL);

	for (i=1; i<(*argc)-1; i++)
	{
		if (!multiple_files){
			// Argument: grid parameter file name.
			if (strcmp("-ffile", argv[i]) == 0)
			{
				ffile_given = 1;
				mypars->fldfile = strdup(argv[i+1]);
			}

			// Argument: ligand pdbqt file name
			if (strcmp("-lfile", argv[i]) == 0)
			{
				lfile_given = 1;
				mypars->ligandfile = strdup(argv[i+1]);
			}
		}

		// Argument: flexible residue pdbqt file name
		if (strcmp("-flexres", argv[i]) == 0)
		{
			mypars->flexresfile = strdup(argv[i+1]);
		}

		// Argument: unbound model to be used.
		// 0 means the bound, 1 means the extended, 2 means the compact ...
		// model's free energy coefficients will be used during docking.
		if (strcmp("-ubmod", argv[i]) == 0)
		{
			sscanf(argv[i+1], "%ld", &tempint);
			switch(tempint){
				case 0:
					mypars->unbound_model = 0;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				case 1:
					mypars->unbound_model = 1;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				case 2:
					mypars->unbound_model = 2;
					mypars->coeffs = unbound_models[mypars->unbound_model];
					break;
				default:
					printf("Warning: value of -ubmod argument ignored. Can only be 0 (unbound same as bound), 1 (extended), or 2 (compact).\n");
			}
		}
	}

	if (ffile_given == 0 && !multiple_files)
	{
		printf("Error: grid fld file was not defined. Use -ffile argument!\n");
		return 1;
	}

	if (lfile_given == 0 && !multiple_files)
	{
		printf("Error: ligand pdbqt file was not defined. Use -lfile argument!\n");
		return 1;
	}

	return 0;
}

int get_commandpars(
                    const int*      argc,
                          char**    argv,
                          double*   spacing,
                          Dockpars* mypars,
                    const bool      late_call
                   )
// The function processes the command line arguments given with the argc and argv parameters,
// and fills the proper fields of mypars according to that. If a parameter was not defined
// in the command line, the default value will be assigned. The mypars' fields will contain
// the data in the same format as it is required for writing it to algorithm defined registers.
{
	int   i;
	int   tempint;
	float tempfloat;
	int   arg_recognized = 0;
	int   arg_set = 1;
	if(late_call){
		// ------------------------------------------
		// default values
		mypars->abs_max_dmov        = 6.0/(*spacing);             // +/-6A
		mypars->base_dmov_mul_sqrt3 = 2.0/(*spacing)*sqrt(3.0);   // 2 A
		mypars->xrayligandfile      = strdup(mypars->ligandfile); // By default xray-ligand file is the same as the randomized input ligand
		if(!mypars->resname){ // only need to set if it's not set yet
			if(strlen(mypars->ligandfile)>6){ // .pdbqt = 6 chars
				i=strlen(mypars->ligandfile)-6;
				mypars->resname = (char*)malloc((i+1)*sizeof(char));
				strncpy(mypars->resname,mypars->ligandfile,i);    // Default is ligand file basename
				mypars->resname[i]='\0';
			} else mypars->resname = strdup("docking");               // Fallback to old default
		}
		// ------------------------------------------
	}

	// overwriting values which were defined as a command line argument
	for (i=1; i<(*argc)-1; i+=2)
	{
		arg_recognized = 0;

		// Argument: number of energy evaluations. Must be a positive integer.
		if (strcmp("-nev", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 0x7FFFFFFF)){
				mypars->num_of_energy_evals = (unsigned long) tempint;
				mypars->nev_provided = true;
			} else
				printf("Warning: value of -nev argument ignored. Value must be between 0 and 2^31-1.\n");
		}

		if (strcmp("-seed", argv[i]) == 0)
		{
			arg_recognized = 1;
			mypars->seed[0] = 0; mypars->seed[1] = 0; mypars->seed[2] = 0;
			tempint = sscanf(argv[i+1], "%u,%u,%u", &(mypars->seed[0]), &(mypars->seed[1]), &(mypars->seed[2]));
		}

		// Argument: number of generations. Must be a positive integer.
		if (strcmp("-ngen", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 16250000))
				mypars->num_of_generations = (unsigned long) tempint;
			else
				printf("Warning: value of -ngen argument ignored. Value must be between 0 and 16250000.\n");
		}

		// Argument: initial sw number of generations. Must be a positive integer.
		if (strcmp("-initswgens", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint >= 0) && (tempint <= 16250000))
				mypars->initial_sw_generations = (unsigned long) tempint;
			else
				printf("Warning: value of -initswgens argument ignored. Value must be between 0 and 16250000.\n");
		}

		// ----------------------------------
		// Argument: Use Heuristics for number of evaluations (can be overwritten with -nev)
		if (strcmp("-heuristics", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->use_heuristics = false;
			else
				mypars->use_heuristics = true;
		}
		// ----------------------------------

		// Argument: Upper limit for heuristics that's reached asymptotically
		if (strcmp("-heurmax", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint <= 1625000000))
				mypars->heuristics_max = (unsigned long) tempint;
			else
				printf("Warning: value of -heurmax argument ignored. Value must be between 1 and 1625000000.\n");
		}

		// Argument: maximal delta movement during mutation. Must be an integer between 1 and 16.
		// N means that the maximal delta movement will be +/- 2^(N-10)*grid spacing Angström.
		if (strcmp("-dmov", argv[i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv[i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 10))
				mypars->abs_max_dmov = tempfloat/(*spacing);
			else
				printf("Warning: value of -dmov argument ignored. Value must be a float between 0 and 10.\n");
		}

		// Argument: maximal delta angle during mutation. Must be an integer between 1 and 17.
		// N means that the maximal delta angle will be +/- 2^(N-8)*180/512 degrees.
		if (strcmp("-dang", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0) && (tempfloat < 180))
				mypars->abs_max_dang = tempfloat;
			else
				printf("Warning: value of -dang argument ignored. Value must be a float between 0 and 180.\n");
		}

		// Argument: mutation rate. Must be a float between 0 and 100.
		// Means the rate of mutations (cca) in percent.
		if (strcmp("-mrat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 100.0))
				mypars->mutation_rate = tempfloat;
			else
				printf("Warning: value of -mrat argument ignored. Value must be a float between 0 and 100.\n");
		}

		// Argument: crossover rate. Must be a float between 0 and 100.
		// Means the rate of crossovers (cca) in percent.
		if (strcmp("-crat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat <= 100.0))
				mypars->crossover_rate = tempfloat;
			else
				printf("Warning: value of -crat argument ignored. Value must be a float between 0 and 100.\n");
		}

		// Argument: local search rate. Must be a float between 0 and 100.
		// Means the rate of local search (cca) in percent.
		if (strcmp("-lsrat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			/*
			if ((tempfloat >= 0.0) && (tempfloat < 100.0))
			*/
			if ((tempfloat >= 0.0) && (tempfloat <= 100.0))
				mypars->lsearch_rate = tempfloat;
			else
				printf("Warning: value of -lrat argument ignored. Value must be a float between 0 and 100.\n");
		}

		// Smoothed pairwise potentials
		if (strcmp("-smooth", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			// smooth is measured in Angstrom
			if ((tempfloat >= 0.0f) && (tempfloat <= 0.5f))
				mypars->smooth = tempfloat;
			else
				printf("Warning: value of -smooth argument ignored. Value must be a float between 0 and 0.5.\n");
		}

		// Argument: local search method:
		// "sw": Solis-Wets
		// "sd": Steepest-Descent
		// "fire": FIRE
		// "ad": ADADELTA
		// "adam": ADAM
		if (strcmp("-lsmet", argv [i]) == 0)
		{
			arg_recognized = 1;

			char* temp = strdup(argv [i+1]);

			if (strcmp(temp, "sw") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 300;
			}
			else if (strcmp(temp, "sd") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "fire") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "ad") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else if (strcmp(temp, "adam") == 0) {
				strcpy(mypars->ls_method, temp);
				//mypars->max_num_of_iters = 30;
			}
			else {
				printf("Error: Value of -lsmet must be a valid string: \"sw\", \"sd\", \"fire\", \"ad\", or \"adam\".\n");
				exit(-1);
			}
			
			free(temp);
		}

		// Argument: tournament rate. Must be a float between 50 and 100.
		// Means the probability that the better entity wins the tournament round during selectin
		if (strcmp("-trat", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= /*5*/0.0) && (tempfloat <= 100.0))
				mypars->tournament_rate = tempfloat;
			else
				printf("Warning: value of -trat argument ignored. Value must be a float between 0 and 100.\n");
		}


		// Argument: rho lower bound. Must be a float between 0 and 1.
		// Means the lower bound of the rho parameter (possible stop condition for local search).
		if (strcmp("-rholb", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 1.0))
				mypars->rho_lower_bound = tempfloat;
			else
				printf("Warning: value of -rholb argument ignored. Value must be a float between 0 and 1.\n");
		}

		// Argument: local search delta movement. Must be a float between 0 and grid spacing*64 A.
		// Means the spread of unifily distributed delta movement of local search.
		if (strcmp("-lsmov", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < (*spacing)*64/sqrt(3.0)))
				mypars->base_dmov_mul_sqrt3 = tempfloat/(*spacing)*sqrt(3.0);
			else
				printf("Warning: value of -lsmov argument ignored. Value must be a float between 0 and %lf.\n", 64*(*spacing));
		}

		// Argument: local search delta angle. Must be a float between 0 and 103°.
		// Means the spread of unifily distributed delta angle of local search.
		if (strcmp("-lsang", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat > 0.0) && (tempfloat < 103.0))
				mypars->base_dang_mul_sqrt3 = tempfloat*sqrt(3.0);
			else
				printf("Warning: value of -lsang argument ignored. Value must be a float between 0 and 103.\n");
		}

		// Argument: consecutive success/failure limit. Must be an integer between 1 and 255.
		// Means the number of consecutive successes/failures after which value of rho have to be doubled/halved.
		if (strcmp("-cslim", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 256))
				mypars->cons_limit = (unsigned long) (tempint);
			else
				printf("Warning: value of -cslim argument ignored. Value must be an integer between 1 and 255.\n");
		}

		// Argument: maximal number of iterations for local search. Must be an integer between 1 and 262143.
		// Means the number of iterations after which the local search algorithm has to terminate.
		if (strcmp("-lsit", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint > 0) && (tempint < 262144))
				mypars->max_num_of_iters = (unsigned long) tempint;
			else
				printf("Warning: value of -lsit argument ignored. Value must be an integer between 1 and 262143.\n");
		}

		// Argument: size of population. Must be an integer between 32 and CPU_MAX_POP_SIZE.
		// Means the size of the population in the genetic algorithm.
		if (strcmp("-psize", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint >= 2) && (tempint <= MAX_POPSIZE))
				mypars->pop_size = (unsigned long) (tempint);
			else
				printf("Warning: value of -psize argument ignored. Value must be an integer between 2 and %d.\n", MAX_POPSIZE);
		}

		// Argument: load initial population from file instead of generating one.
		// If the value is zero, the initial population will be generated randomly, otherwise it will be loaded from a file.
		if (strcmp("-pload", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->initpop_gen_or_loadfile = false;
			else
				mypars->initpop_gen_or_loadfile = true;
		}

		// Argument: number of pdb files to be generated.
		// The files will include the best docking poses from the final population.
		if (strcmp("-npdb", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint < 0) || (tempint > MAX_POPSIZE))
				printf("Warning: value of -npdb argument ignored. Value must be an integer between 0 and %d.\n", MAX_POPSIZE);
			else
				mypars->gen_pdbs = tempint;
		}

		// ---------------------------------
		// UPDATED in : get_filelist()
		// ---------------------------------
		// Argument: name of file containing file list
		if (strcmp("-filelist", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// UPDATED in : preparse_dpf()
		// ---------------------------------
		// Argument: name of file containing file list
		if (strcmp("-import_dpf", argv [i]) == 0)
			arg_recognized = 1;

		// ---------------------------------
		// MISSING: char* fldfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of grid parameter file.
		if (strcmp("-ffile", argv [i]) == 0){
			arg_recognized = 1;
			arg_set = 0;
		}

		// ---------------------------------
		// MISSING: char* ligandfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of ligand pdbqt file
		if (strcmp("-lfile", argv [i]) == 0){
			arg_recognized = 1;
			arg_set = 0;
		}

		// ---------------------------------
		// MISSING: char* flexresfile
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: name of ligand pdbqt file
		if (strcmp("-flexres", argv [i]) == 0){
			arg_recognized = 1;
			arg_set = 0;
		}

		// Argument: derivate atom types
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (strcmp("-derivtype", argv [i]) == 0)
		{
			arg_recognized = 1;
		}

		// Argument: modify pairwise atom type parameters (LJ only at this point)
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (strcmp("-modpair", argv [i]) == 0)
		{
			arg_recognized = 1;
		}

		// ---------------------------------
		// MISSING: devnum
		// UPDATED in : main
		// ----------------------------------
		// Argument: OpenCL/Cuda device number to use
		if (strcmp("-devnum", argv [i]) == 0)
		{
			arg_recognized = 1;
			arg_set = 0;
			if(!late_call){
				arg_set = 1;
				sscanf(argv [i+1], "%d", &tempint);
				if ((tempint >= 1) && (tempint <= 65536)){
					mypars->devnum = (unsigned long) tempint-1;
				} else printf("Warning: value of -devnum argument ignored. Value must be an integer between 1 and 65536.\n");
			}
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Multiple CG-G0 maps or not
		// - has already been tested for in
		//   main.cpp, as it's needed at grid
		//   creation time not after (now)
		if (strcmp("-cgmaps", argv [i]) == 0)
		{
			arg_recognized = 1; // stub to not complain about an unknown parameter
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Automatic stopping criterion (1) or not (0)
		if (strcmp("-autostop", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->autostop = 0;
			else
				mypars->autostop = 1;
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Test frequency for auto-stopping criterion
		if (strcmp("-asfreq", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);
			if ((tempint >= 1) && (tempint <= 100))
				mypars->as_frequency = (unsigned int) tempint;
			else
				printf("Warning: value of -asfreq argument ignored. Value must be an integer between 1 and 100.\n");
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Stopping criterion standard deviation.. Must be a float between 0.01 and 2.0;
		// Means the energy standard deviation of the best candidates after which to stop evaluation when autostop is 1..
		if (strcmp("-stopstd", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.01) && (tempfloat < 2.0))
				mypars->stopstd = tempfloat;
			else
				printf("Warning: value of -stopstd argument ignored. Value must be a float between 0.01 and 2.0.\n");
		}
		// ----------------------------------

		// ----------------------------------
		// Argument: Minimum electrostatic pair potential distance .. Must be a float between 0.0 and 2.0;
		// This will cut the electrostatics interaction to the value at that distance below it. (default: 0.01)
		if (strcmp("-elecmindist", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if ((tempfloat >= 0.0) && (tempfloat < 2.0))
				mypars->elec_min_distance = tempfloat;
			else
				printf("Warning: value of -elecmindist argument ignored. Value must be a float between 0.0 and 2.0.\n");
		}
		// ----------------------------------

		// Argument: number of runs. Must be an integer between 1 and 1000.
		// Means the number of required runs
		if (strcmp("-nrun", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS))
				mypars->num_of_runs = (int) tempint;
			else
				printf("Warning: value of -nrun argument ignored. Value must be an integer between 1 and %d.\n", MAX_NUM_OF_RUNS);
		}

		// Argument: energies of reference ligand required.
		// If the value is not zero, energy values of the reference ligand is required.
		if (strcmp("-rlige", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->reflig_en_required = false;
			else
				mypars->reflig_en_required = true;
		}

		// ---------------------------------
		// MISSING: char unbound_model
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		// Argument: unbound model to be used.
		if (strcmp("-ubmod", argv [i]) == 0){
			arg_recognized = 1;
			arg_set = 0;
		}

		// Argument: handle molecular symmetry during rmsd calculation
		// If the value is not zero, molecular syymetry will be taken into account during rmsd calculation and clustering.
		if (strcmp("-hsym", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->handle_symmetry = false;
			else
				mypars->handle_symmetry = true;
		}

		// Argument: generate final population result files.
		// If the value is zero, result files containing the final populations won't be generated, otherwise they will.
		if (strcmp("-gfpop", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->gen_finalpop = false;
			else
				mypars->gen_finalpop = true;
		}

		// Argument: generate best.pdbqt
		// If the value is zero, best.pdbqt file containing the coordinates of the best result found during all of the runs won't be generated, otherwise it will
		if (strcmp("-gbest", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->gen_best = false;
			else
				mypars->gen_best = true;
		}

		// Argument: name of result files.
		if (strcmp("-resnam", argv [i]) == 0)
		{
			arg_recognized = 1;
			free(mypars->resname); // as we assign a default value dynamically created to it
			mypars->resname = strdup(argv [i+1]);
		}

		// Argument: use modified QASP (from VirtualDrug) instead of original one used by AutoDock
		// If the value is not zero, the modified parameter will be used.
		if (strcmp("-modqp", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);

			if (tempint == 0)
				mypars->qasp = 0.01097f; // original AutoDock QASP parameter
			else
				mypars->qasp = 0.00679f; // from VirtualDrug
		}

		// Argument: rmsd tolerance for clustering.
		// This will be used during clustering for the tolerance distance.
		if (strcmp("-rmstol", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%f", &tempfloat);

			if (tempfloat > 0.0)
				mypars->rmsd_tolerance = tempfloat;
			else
				printf("Warning: value of -rmstol argument ignored. Value must be a double greater than 0.\n");
		}

		// Argument: choose wether to output XML or not
		// If the value is 1, XML output will still be generated
		// XML output won't be generated if 0 is specified
		if (strcmp("-xmloutput", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%d", &tempint);
			
			if (tempint == 0)
				mypars->output_xml = false;
			else
				mypars->output_xml = true;
		}

		// ----------------------------------
		// Argument: ligand xray pdbqt file name
		if (strcmp("-xraylfile", argv[i]) == 0)
		{
			arg_recognized = 1;
			free(mypars->xrayligandfile);
			mypars->xrayligandfile = strdup(argv[i+1]);
			mypars->given_xrayligandfile = true;
			printf("Info: using -xraylfile value as X-ray ligand.\n");
		}
		// ----------------------------------


		if (arg_recognized != 1){
			printf("Warning: ignoring unknown argument '%s'.\n", argv [i]);
		}
	}

	// validating some settings

	if (mypars->pop_size < mypars->gen_pdbs)
	{
		printf("Warning: value of -npdb argument ignored. Value cannot be greater than the population size.\n");
		mypars->gen_pdbs = 1;
	}
	
	return arg_recognized + (arg_set<<1);
}

void gen_initpop_and_reflig(
                                  Dockpars*   mypars,
                                  float*      init_populations,
                                  float*      ref_ori_angles,
                                  Liganddata* myligand,
                            const Gridinfo*   mygrid
                           )
// The function generates a random initial population
// (or alternatively, it reads from an external file according to mypars),
// and the angles of the reference orientation.
// The parameters mypars, myligand and mygrid describe the current docking.
// The pointers init_population and ref_ori_angles have to point to
// two allocated memory regions with proper size which the function will fill with random values.
// Each contiguous GENOTYPE_LENGTH_IN_GLOBMEM pieces of floats in init_population corresponds to a genotype,
// and each contiguous three pieces of floats in ref_ori_angles corresponds to
// the phi, theta and angle genes of the reference orientation.
// In addition, as part of reference orientation handling,
// the function moves myligand to origo and scales it according to grid spacing.
{
	int entity_id, gene_id;
	int gen_pop, gen_seeds;
	FILE* fp;
	int i;
	float init_orientation[MAX_NUM_OF_ROTBONDS+6];
	double movvec_to_origo[3];

	int pop_size = mypars->pop_size;

	float u1, u2, u3; // to generate random quaternion
	float qw, qx, qy, qz; // random quaternion
	float x, y, z, s; // convert quaternion to angles
	float phi, theta, rotangle;

	// initial population
	gen_pop = 0;

	// Reading initial population from file if only 1 run was requested
	if (mypars->initpop_gen_or_loadfile)
	{
		if (mypars->num_of_runs != 1)
		{
			printf("Warning: more than 1 run was requested. New populations will be generated \ninstead of being loaded from initpop.txt\n");
			gen_pop = 1;
		}
		else
		{
			fp = fopen("initpop.txt","rb"); // fp = fopen("initpop.txt","r");
			if (fp == NULL)
			{
				printf("Warning: can't find initpop.txt. A new population will be generated.\n");
				gen_pop = 1;
			}
			else
			{
				for (entity_id=0; entity_id<pop_size; entity_id++)
					for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
						fscanf(fp, "%f", &(init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]));

				// reading reference orienation angles from file
				fscanf(fp, "%f", &(mypars->ref_ori_angles[0]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[1]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[2]));

				fclose(fp);
			}
		}
	}
	else
		gen_pop = 1;

	// Local random numbers for thread safety/reproducibility
	LocalRNG r(mypars->seed);

	// Generating initial population
	if (gen_pop == 1)
	{
		for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++)
		{
			for (gene_id=0; gene_id<3; gene_id++)
			{
				init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*(mygrid->size_xyz_angstr[gene_id]);
			}
			// generate random quaternion
			u1 = r.random_float();
			u2 = r.random_float();
			u3 = r.random_float();
			qw = sqrt(1.0 - u1) * sin(PI_TIMES_2 * u2);
			qx = sqrt(1.0 - u1) * cos(PI_TIMES_2 * u2);
			qy = sqrt(      u1) * sin(PI_TIMES_2 * u3);
			qz = sqrt(      u1) * cos(PI_TIMES_2 * u3);

			// convert to angle representation
			rotangle = 2.0 * acos(qw);
			s = sqrt(1.0 - (qw * qw));
			if (s < 0.001){ // rotangle too small
				x = qx;
				y = qy;
				z = qz;
			} else {
				x = qx / s;
				y = qy / s;
				z = qz / s;
			}

			theta = acos(z);
			phi = atan2(y, x);

			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = phi / DEG_TO_RAD;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = theta / DEG_TO_RAD;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = rotangle / DEG_TO_RAD;

			//printf("angles = %8.2f, %8.2f, %8.2f\n", phi / DEG_TO_RAD, theta / DEG_TO_RAD, rotangle/DEG_TO_RAD);

			/*
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) myrand() * 360;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) myrand() * 180;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) myrand() * 360;
			*/

			for (gene_id=6; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++) {
				init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*360;
			}
		}

		// Writing first initial population to initpop.txt
		fp = fopen("initpop.txt", "w");
		if (fp == NULL)
			printf("Warning: can't create initpop.txt.\n");
		else
		{
			for (entity_id=0; entity_id<pop_size; entity_id++)
				for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
					fprintf(fp, "%f ", init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]);

			// writing reference orientation angles to initpop.txt
			fprintf(fp, "%f ", mypars->ref_ori_angles[0]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[1]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[2]);

			fclose(fp);
		}
	}

	// genotypes should contain x, y and z genes in grid spacing instead of Angstroms
	// (but was previously generated in Angstroms since fdock does the same)

	for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++)
		for (gene_id=0; gene_id<3; gene_id++)
			init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]/mygrid->spacing;

	// changing initial orientation of reference ligand
	/*for (i=0; i<38; i++)
		switch (i)
		{
		case 3: init_orientation [i] = mypars->ref_ori_angles [0];
				break;
		case 4: init_orientation [i] = mypars->ref_ori_angles [1];
				break;
		case 5: init_orientation [i] = mypars->ref_ori_angles [2];
				break;
		default: init_orientation [i] = 0;
		}

	change_conform_f(myligand, init_orientation, 0);*/

	// initial orientation will be calculated during docking,
	// only the required angles are generated here,
	// but the angles possibly read from file are ignored

	for (i=0; i<mypars->num_of_runs; i++)
	{
		// uniform distr.
		// generate random quaternion
		u1 = r.random_float();
		u2 = r.random_float();
		u3 = r.random_float();
		qw = sqrt(1.0 - u1) * sin(PI_TIMES_2 * u2);
		qx = sqrt(1.0 - u1) * cos(PI_TIMES_2 * u2);
		qy = sqrt(      u1) * sin(PI_TIMES_2 * u3);
		qz = sqrt(      u1) * cos(PI_TIMES_2 * u3);

		// convert to angle representation
		rotangle = 2.0 * acos(qw);
		s = sqrt(1.0 - (qw * qw));
		if (s < 0.001){ // rotangle too small
			x = qx;
			y = qy;
			z = qz;
		} else {
			x = qx / s;
			y = qy / s;
			z = qz / s;
		}

		theta = acos(z);
		phi = atan2(y, x);

		ref_ori_angles[3*i]   = phi / DEG_TO_RAD;
		ref_ori_angles[3*i+1] = theta / DEG_TO_RAD;
		ref_ori_angles[3*i+2] = rotangle / DEG_TO_RAD;
	}

	get_movvec_to_origo(myligand, movvec_to_origo);
	double flex_vec[3];
	for (unsigned int i=0; i<3; i++)
		flex_vec [i] = -mygrid->origo_real_xyz [i];
	move_ligand(myligand, movvec_to_origo, flex_vec);
	scale_ligand(myligand, 1.0/mygrid->spacing);
	get_moving_and_unit_vectors(myligand);

	/*
	printf("ligand: movvec_to_origo: %f %f %f\n", movvec_to_origo[0], movvec_to_origo[1], movvec_to_origo[2]);
	*/

}
