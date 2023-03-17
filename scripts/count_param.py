
from improved_diffusion.script_util import get_config, create_model
from improved_diffusion.models.nn_utils import count_parameters
from improved_diffusion import dist_util

def main():
    config, config_override = get_config() # read config yaml at --config
    dist_util.setup_dist()
    
    '''
    setup model
    '''
    print("creating model and diffusion...")
    model = create_model(config)

    print('number of parameters: ', count_parameters(model))
    print(model)


if __name__ == "__main__":
    main()