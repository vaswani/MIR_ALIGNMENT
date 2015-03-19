#pragma once
using namespace std;
#include <math.h>
#include "dualDecomp.h"

typedef vector<vector<pair<unsigned int,unsigned int> > > word_id_map;
class coupledReg {
   public: 
     word_id_map ef_map;
     int Model1_Iterations;
     int HMM_Iterations;
     int total_ef_pairs;
     int total_fe_pairs;
     vcbList *eTrainVcbList, *fTrainVcbList;
     float lambda_ef;
     float lambda_fe;
     float reg_lambda;
 
  coupledReg(int model1_iter,
      int hmm_iter,
      int ef_pairs,
      int fe_pairs,
      vcbList *eTrainVcbList,
      vcbList *fTrainVcbList,
      float lambda_ef,
      float lambda_fe,
      float reg_lambda):
    ef_map(word_id_map ()),
    Model1_Iterations(model1_iter),
    HMM_Iterations(hmm_iter),
    total_ef_pairs(ef_pairs),
    total_fe_pairs(fe_pairs),
    eTrainVcbList(eTrainVcbList),
    fTrainVcbList(fTrainVcbList),
    lambda_ef(lambda_ef),
    lambda_fe(lambda_fe),
    reg_lambda(reg_lambda){}

  template<class alignment_model> 
  void buildEFMap(alignment_model &ef_model,alignment_model &fe_model) {
    cerr<<"We are building an ef map inside coupled reg"<<endl;
    ef_model.getTtable().buildEFMap(ef_map,fe_model.getTtable().getLexmat());
  }

  const word_id_map & getEFMap() {
    return ef_map;
  }
  
  template<class alignment_model>
  void runModelCoupledIter(bool seedModel1,
      bool useDict,
      Dictionary *dictionary,
      model_type current_model,
      alignment_model &ef_model,
      alignment_model &fe_model,
      int num_Iterations){
    int ef_minIter,fe_minIter;
    //cerr<<"In runModelCoupledIter, reg lambda was "<<reg_lambda<<endl;
   // dualDecomp dual_decomp_optimizer;
    // RUNNING THE MODELS IN BOTH DIRECTOINS, ONE ITERATION AT A TIME
    /*
    float current_reg_term_value = regFuncValue(
       ef_current_points,
       fe_current_points,
       ef_rowwiseExpCntsSum,
       ef_map,
       reg_lambda,
       eTrainVcbList,
       fTrainVcbList,
       reg_option,
       reg_func_option) ;
    */
    float current_reg_term_value = 0; 
    float current_objective = 0;
    for (int it=1; it<=num_Iterations; it++) {
      vector<Array<double> > ef_posteriors;
      vector<Array<double> > fe_posteriors;
      //cout<<" Running regular model 1 for iteration "<<it<<endl;
      //minIter=m1.em_with_tricks_e_step(it,seedModel1,*dictionary, useDict);
      // THE SIGNATURES OF THE HMM E-STEP AND MODEL1 E-STEP ARE DIFFERENT. 
      // I THEREFORE HAVE TO KNOW WHICH MODEL IT IS
      if (current_model == M1) {
        cout<<" Running e given f model1  for iteration "<<it<<endl;
        ef_minIter=ef_model.em_with_tricks_e_step(it,seedModel1,dictionary, useDict,"ef",ef_posteriors);
        
        cout<<" Running f given e model2 for iteration "<<it<<endl;
        fe_minIter=fe_model.em_with_tricks_e_step(it,seedModel1,dictionary, useDict,"fe",fe_posteriors);
      }

      if (current_model == HMM) {
        cout<<" Running e given f hmm model for iteration "<<it<<endl;
        ef_minIter=ef_model.em_with_tricks_e_step(it,false,NULL,false,"ef",ef_posteriors);
        cout<<" Running f given e hmm model for iteration "<<it<<endl;
        fe_minIter=fe_model.em_with_tricks_e_step(it,false,NULL,false,"fe",fe_posteriors);
        //cout<<"The ef posteriors are "<<ef_posteriors<<endl;
        //cout<<"The fe posteriors are "<<fe_posteriors<<endl;
        //Write the posteriors
        cerr<<"Size of ef posteriors is "<<ef_posteriors.size()<<endl;
        cerr<<"Size of fe posteriors is "<<fe_posteriors.size()<<endl;
        if (ef_posteriors.size() != fe_posteriors.size()) {
          cerr<<"WARNING. ef and fe posteriors had different sizes "<<endl;
        }
        writePosteriors(it,"ef",ef_posteriors);
        writePosteriors(it,"fe",fe_posteriors);
      }

      vector<vector<float> > ef_optimizedProbs,fe_optimizedProbs;

      cerr<<"The total objective function value in iteration "<<it<<" was "<<-total_ef_pairs*log(ef_model.getPerp()) -total_fe_pairs*log(fe_model.getPerp())-current_reg_term_value<<endl;
      //RUNNING THE PGD M STEP for ef_model and fe_model
      if ((current_model == M1 && it >= 2) || (current_model == HMM)) {
        if (pgd_flag == 1) {
          current_reg_term_value = runPGDMStep(
              ef_model,
              fe_model,
              lambda_ef,
              lambda_fe,
              reg_lambda,
              *eTrainVcbList,
              *fTrainVcbList,
              ef_map);
        }
        /*
        if (dual_decomp_flag == 1) {
          current_reg_term_value = dualDecomp::runDDMStep(
            ef_model,
            fe_model,
            reg_lambda,
            *eTrainVcbList,
            *fTrainVcbList,
            ef_map);
        }
        */
        cerr<<"The current reg term value was "<<current_reg_term_value<<endl;
      } else {
        ef_model.normalizeTable();
        fe_model.normalizeTable();
      }
      ef_model.printTableAndReport(it,"ef");
      fe_model.printTableAndReport(it,"fe");

      // Assgining the optimized probabilities
      
    }

  }

  void writePosteriors(int iter,
      string direction,
      vector<Array<double> > &posteriors) {
    stringstream filename;
    filename<<"posteriors."<<direction<<"."<<iter;
    ofstream posterior_file;
    posterior_file.open (filename.str().c_str());
    for (int i=0;i<posteriors.size()-1;i++){
      for (int j=0;j<posteriors[i].size()-1;j++) {
        posterior_file<<posteriors[i][j]<<",";
      }
      posterior_file<<posteriors[i][posteriors[i].size()-1]<<endl;
    }
    //printing the last posteriors
    unsigned int last_index = posteriors.size()-1;
    for (int j=0;j<posteriors[last_index].size()-1;j++) {
      posterior_file<<posteriors[last_index][j]<<",";
    }
    posterior_file<<posteriors[last_index][posteriors[last_index].size()-1]<<endl;
    posterior_file.close();
  }
  
  template<class alignment_model>
  static float runPGDMStep(
      alignment_model &ef_model,
      alignment_model &fe_model,
      const float lambda_ef,
      const float lambda_fe,
      const float reg_lambda,
      vcbList &eTrainVcbList,
      vcbList &fTrainVcbList,
      const word_id_map &ef_map) {
    cout<<"Running PGD m-step"<<endl;
    cout<<"lambda ef is "<<lambda_ef<<endl;
    cout<<"lambda fe is "<<lambda_fe<<endl;
    //cerr<<"The reg lambda was "<<reg_lambda<<endl;
    //running the PGD m step
    // STORING EXPECTED COUNTS
    vector<vector<float> > ef_expCntsVec,ef_probsVec;
    vector<vector<float> > fe_expCntsVec,fe_probsVec;
    vector<vector<float> > ef_optimizedProbs,fe_optimizedProbs;
    vector<float> ef_rowwiseExpCntsSum,fe_rowwiseExpCntsSum;
    cout<<" ACCUMULATING EXPECTED COUNTS FROM E given F"<<endl;
    ef_model.getTtable().getCounts(&ef_expCntsVec,&ef_rowwiseExpCntsSum);
    //printCounts(ef_expCntsVec);
    //getchar();
    cout<<" ACCUMULATING PROBABILITIES FROM E GIVEN F"<<endl;
    ef_model.getTtable().getProbs(&ef_probsVec);
    //cout<<"Printing the ef probs"<<endl;
    //printCounts(ef_probsVec);
    //getchar();
    cout<<" ACCUMULATING EXPECTED COUNTS FROM E given F"<<endl;
    fe_model.getTtable().getCounts(&fe_expCntsVec,&fe_rowwiseExpCntsSum);
    //printCounts(ef_expCntsVec);
    //getchar();

    cout<<" ACCUMULATING PROBABILITIES FROM E GIVEN F"<<endl;
    fe_model.getTtable().getProbs(&fe_probsVec);
    //printCounts(fe_probsVec);
    //getchar();
    //now normalize table
    enum regularization_type reg_option;
    enum regularization_func_type reg_func_option;
    // Contitional reg and joint reg cannot both be 1. 
    // xor gives us that
    if (conditional_reg^joint_reg == 0){
      cerr<<"Warning..."<<endl;
      cerr<<"Ivalid options for regularization"<<endl;
      cerr<<"Setting joint reg to 1"<<endl;
      conditional_reg = 0;
      joint_reg = 1;
    }
    if (l1_reg^l2_reg == 0){
      cerr<<"Warning..."<<endl;
      cerr<<"Ivalid options for regularization type(l1 or l2)"<<endl;
      cerr<<"Setting conditional reg to l2"<<endl;
      l2_reg = 1;
      l1_reg = 0;
    }
    //assert((conditional_reg^joint_reg) != 0 );
    // Same for l2 reg option and l1 reg option
    //assert((l1_reg^l2_reg) != 0);
    if (conditional_reg == 1) {
      reg_option = CONDITIONAL;
      cerr<<"The reg func option was CONDITIONAL"<<endl;
    }
    if (joint_reg == 1) {
      reg_option = JOINT;
      cerr<<"The reg option was group JOINT"<<endl;
    }
    if (l1_reg == 1) {
      reg_func_option = L1;
      cerr<<"The reg func option was L1"<<endl;
    }
    if (l2_reg == 1) {
      reg_func_option = L2;
      cerr<<"The reg func option was L2 reg"<<endl;
    }
    if (group_lasso ==1) {
      reg_func_option = GLASSO;
      cerr<<"The reg func option was group lasso"<<endl;
    }
    if (product==1) {
      reg_option = PRODUCT;
      cerr<<"The reg option was product"<<endl;
    }
    if (product_sqrt==1) {
      reg_option = PRODUCT_SQRT;
      cerr<<"The reg option was prodct sqrt"<<endl;
    }

    float reg_term_value = 0.;
    cerr<<"The pgd flag was "<<pgd_flag;
    if (pgd_flag == 1) {
      reg_term_value = projectedGradientDescentWithArmijoRule(
        ef_expCntsVec,
        ef_probsVec,
        ef_rowwiseExpCntsSum,
        ef_optimizedProbs,
        fe_expCntsVec,
        fe_probsVec,
        fe_rowwiseExpCntsSum,
        fe_optimizedProbs,
        lambda_ef,
        lambda_fe,
        reg_lambda,
        ef_map,
        eTrainVcbList,
        fTrainVcbList,
        reg_option,
        reg_func_option);
    }
    /*
    cerr<<"the exponentiated gradient flag was "<<exponentiated_gradient_flag<<endl;
    if (exponentiated_gradient_flag == 1) {
      reg_term_value = exponentiatedGradient(
          ef_expCntsVec,
          ef_probsVec,
          ef_rowwiseExpCntsSum,
          ef_optimizedProbs,
          fe_expCntsVec,
          fe_probsVec,
          fe_rowwiseExpCntsSum,
          fe_optimizedProbs,
          reg_lambda,
          ef_map,
          eTrainVcbList,
          fTrainVcbList,
          reg_option,
          reg_func_option);
    }
    */

    // After running PGD, we need to assign probs into the
    // t-table
    /*
    cout<<"printing optimized probs after pgd. The size is"<<ef_optimizedProbs.size()<<endl;
    for (int i=0;i<ef_optimizedProbs.size();i++) {
      cout<<"ith vector is "<<endl;
      printVector(ef_optimizedProbs[i]);
    }
    getchar(); 
    */
    ef_model.getMutableTtable().setProbs(ef_optimizedProbs);
    fe_model.getMutableTtable().setProbs(fe_optimizedProbs);
    return(reg_term_value);
  }

};
