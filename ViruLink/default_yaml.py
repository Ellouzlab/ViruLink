# If a database config yaml is not provided
# this will be used as the default configuration.
import yaml

default_yaml_dct = yaml.safe_load('''
name: Default
description: Default configuration for ViruLink
modes: [normal]

# Parameters for modes
settings:

  # if the --fast flag is not used
  normal:

    #Configs for graph making
    graph_making:
      ANI:
        # Either skani or mmseqs
        ani_program: skani
        minimum_ani: 0.2
        consider_alignment_length: true
        alpha: 2
      
      hypergeometric:
        # Protein search parameters
        e_value: 0.00001
        percent_id: 60
        db_cov: 50
        bitscore: 50
        
        # Either vogdb or custom_prot_db
        # Depends on if there is a custom protein database provided
        protein_db: custom_prot_db

        # Calculation of edges
        weight_type: ratio
        hypergeom_pval: 0.1
    
    #Configs for node2vec
    n2v:
      walk_length: 10
      
      # High p and low q values => better species/genus level classification
      # Low p and high q values => better family level + level classification
      p: 0.5
      q: 2
      
      walks_per_node: 200
      window_size: 5
      epochs: 5
      embedding_dim: 128
    
    training_params:
      # RNG_seed only changes test/train/val splits. Not the random walk, nor CUDA's non-determinism.
      # As such, you will still get different results if you run the same model with the same RNG_seed.
      RNG_seed: 42
      EPOCHS: 10
      BATCH: 512
      LEARNING_RATE: 0.001
      TRIANGLES_PER_CLASS_train: 8000
      TRIANGLES_PER_CLASS_eval: 2000
      
      Model:
        Activation: swiglu
        lambda_int: 1.0
        lambda_tri: 0.5
        max_recycles: 3
        gate_alpha: 0.12
        mono_lambda: 0.10
        aux_loss_weight: 0.5
        
        # Attention heads have huge impact on the model's performance.
        # More heads => more parameters => more memory usage.
        # More heads => better performance
        attn_heads: 4
        attn_layers: 2
        attn_dropout: 0.1
        edge_feature_dim_refiner: 32 
        early_stopping_patience: 5
        evaluation_metrics_enabled: true
        rescale_ani_weights: false

      
      # If you've decided to change the levels of classification
      # you can do so here. Just remember, ViruLink learns from the available data.
      # As such, look inside the csv files of your database of interest. If your level
      # of interest is not present in the columns AND defined for at least a quarter of
      # the samples, with a bunch of variety, then ViruLink will not learn it.
      Levels:
        NR: true
        Realm: false
        Kingdom: false
        Subkingdom: false
        Phylum: false
        Subphylum: false
        Subclass: false
        Class: false
        Subclass: false
        Order: false
        Suborder: false
        Family: true
        Subfamily: true
        Genus: true
        Subgenus: false
        Species: false
        Isolate: false
    ''')

      

        


