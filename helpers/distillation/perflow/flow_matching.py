import os, logging
import torch
import torch.nn.functional as F
from helpers.distillation.common import DistillationBase
from helpers.training.custom_schedule import TimeWindows

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

class FlowMatchingPeRFlowDistiller(DistillationBase):
    """Implementation of PeRFlow distillation adapted for flow matching models."""
    
    def __init__(self, teacher_model, student_model=None, config=None):
        # FlowMatching PeRFlow specific config defaults
        flow_perflow_config = {
            'loss_type': 'velocity_matching',
            'solving_steps': 2,
            'windows': 16,
            'support_cfg': False,
            'cfg_sync': True, 
            'discrete_timesteps': -1,
        }
        
        # Update with user config
        if config:
            flow_perflow_config.update(config)
            
        super().__init__(teacher_model, student_model, flow_perflow_config)
        
        # Check if the model is flow matching
        if not self.is_flow_matching:
            raise ValueError("Teacher model must be a flow matching model for FlowMatchingPeRFlowDistiller")
        
        # Initialize custom components for flow matching
        self._init_flow_matching_components()
    
    def _init_flow_matching_components(self):
        """Initialize components needed for flow matching PeRFlow."""
        # Create a time windows lookup mechanism for flow matching
        self._create_time_windows()
        
        # Store reference to scheduler
        self.flow_scheduler = self.teacher_scheduler
    
    def _create_time_windows(self):
        """Create time windows for flow matching."""        
        # Create and store time windows instance
        self.time_windows = TimeWindows(num_windows=self.config['windows'])
    
    def solve_flow(self, prepared_batch, t_start, t_end, guidance_scale=1.0):
        """
        Solve the flow ODE from t_start to t_end using the teacher model.
        
        Args:
            prepared_batch: The complete prepared batch with latents and embeddings
            t_start: Starting time
            t_end: Ending time
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            The solved latents at t_end
        """
        # Debugging the shape and values of inputs
        logger.info(f"Solving flow with t_start shape: {t_start.shape}, t_end shape: {t_end.shape}")
        
        # Get inputs from the prepared batch
        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get("negative_encoder_hidden_states")
        
        # Log shapes for debugging
        logger.info(f"latents shape: {latents.shape}, prompt_embeds shape: {prompt_embeds.shape}")
        if negative_prompt_embeds is not None:
            logger.info(f"negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
        
        device = latents.device
        dtype = latents.dtype
        
        # Prepare for guidance
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
        logger.info(f"Using classifier-free guidance: {do_classifier_free_guidance}")
        
        # Solve using multiple steps
        num_steps = self.config['solving_steps']
        
        # Handle different tensor dimensions for step_size
        if len(latents.shape) == 5:  # Video model (B, C, F, H, W)
            step_size = (t_start - t_end) / num_steps
        else:  # Image model (B, C, H, W)
            step_size = (t_start - t_end) / num_steps
        
        # Current latents
        current_latents = latents.clone()
        current_t = t_start.clone()
        
        for i in range(num_steps):
            logger.debug(f"Solving step {i+1}/{num_steps}")
            
            # Create a new model_inputs dict with only the necessary inputs
            model_inputs = prepared_batch.copy()
            model_inputs.update({
                "latents": current_latents,
                "noisy_latents": current_latents,  # For flow models, these are the same
                "timesteps": current_t,
                "encoder_hidden_states": prompt_embeds,
            })
            
            # Add any additional required inputs from the prepared batch
            for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                if key in prepared_batch:
                    model_inputs[key] = prepared_batch[key]
            
            if do_classifier_free_guidance:
                # For guidance, we need to run the model twice and combine the results
                
                # Create conditional inputs
                cond_inputs = model_inputs.copy()
                
                # Create unconditional inputs - duplicate the latents and timesteps
                uncond_inputs = {
                    "latents": current_latents,
                    "noisy_latents": current_latents,
                    "timesteps": current_t,
                    "encoder_hidden_states": negative_prompt_embeds,
                }
                
                # Add any additional required inputs
                for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                    if key in prepared_batch:
                        uncond_inputs[key] = prepared_batch[key]
                
                # Get unconditional prediction
                with torch.no_grad():
                    uncond_prediction = self.teacher_model.model_predict(uncond_inputs)["model_prediction"]
                
                # Get conditional prediction
                with torch.no_grad():
                    cond_prediction = self.teacher_model.model_predict(cond_inputs)["model_prediction"]
                
                # Apply classifier-free guidance
                model_prediction = uncond_prediction + guidance_scale * (cond_prediction - uncond_prediction)
            else:
                # Get prediction without guidance
                with torch.no_grad():
                    model_prediction = self.teacher_model.model_predict(model_inputs)["model_prediction"]
            
            # Euler step - handle different tensor dimensions
            if len(current_latents.shape) == 5:  # Video model (B, C, F, H, W)
                current_latents = current_latents + step_size[:, None, None, None, None] * model_prediction
                current_t = current_t - step_size
            else:  # Image model (B, C, H, W)
                current_latents = current_latents + step_size[:, None, None, None] * model_prediction
                current_t = current_t - step_size
        
        return current_latents
    
    def prepare_batch(self, batch, model, state):
        """Prepare batch with flow matching PeRFlow-specific processing."""
        # We're doing distillation, not regularization
        # Remove any is_regularisation_data flag that might be set
        if "is_regularisation_data" in batch:
            del batch["is_regularisation_data"]
        
        # If it's a fresh batch (not already prepared by model), prepare it first
        if "noisy_latents" not in batch:
            prepared_batch = model.prepare_batch(batch, state)
        else:
            prepared_batch = batch
            
        # Log what's in the prepared batch for debugging
        logger.info(f"Prepared batch keys: {prepared_batch.keys()}")
        if "latents" in prepared_batch:
            logger.info(f"latents shape: {prepared_batch['latents'].shape}")
        if "noisy_latents" in prepared_batch:
            logger.info(f"noisy_latents shape: {prepared_batch['noisy_latents'].shape}")
        
        # Add flow matching specific processing
        bsz = prepared_batch["latents"].shape[0]
        device = prepared_batch["latents"].device
        
        with torch.no_grad():
            # Sample timepoints uniformly in [0, 1)
            timepoints = torch.rand((bsz,), device=device)
            
            if self.config['discrete_timesteps'] != -1:
                # Discretize timepoints if requested
                timepoints = (timepoints * self.config['discrete_timesteps']).floor() / self.config['discrete_timesteps']
            
            # Convert to [1, 0]
            timepoints = 1 - timepoints
            prepared_batch["perflow_timepoints"] = timepoints
            
            # Get time window endpoints
            t_start, t_end = self.time_windows.lookup_window(timepoints)
            prepared_batch["perflow_t_start"] = t_start
            prepared_batch["perflow_t_end"] = t_end
            
            # For flow matching, the noisy latents are the starting point
            prepared_batch["perflow_latents_start"] = prepared_batch["noisy_latents"].clone()
            
            # Temporarily disable adapter for teacher predictions if using same model
            if self.low_rank_distillation:
                self.toggle_adapter(enable=False)
            
            # Get guidance scale based on config
            guidance_scale = 1.0 if self.config['cfg_sync'] else 7.5
            
            # Solve the flow to get latents at end time
            latents_end = self.solve_flow(
                prepared_batch=prepared_batch,
                t_start=t_start,
                t_end=t_end,
                guidance_scale=guidance_scale,
            )
            
            # Re-enable adapter if using same model
            if self.low_rank_distillation:
                self.toggle_adapter(enable=True)
                
            prepared_batch["perflow_latents_end"] = latents_end
            
            # Interpolate to get latents at the sampled timepoint
            if len(prepared_batch["latents"].shape) == 5:  # Video model (B, C, F, H, W)
                latents_t = prepared_batch["perflow_latents_start"] + (latents_end - prepared_batch["perflow_latents_start"]) / (
                    t_end[:, None, None, None, None] - t_start[:, None, None, None, None]
                ) * (timepoints[:, None, None, None, None] - t_start[:, None, None, None, None])
            else:  # Image model (B, C, H, W)
                latents_t = prepared_batch["perflow_latents_start"] + (latents_end - prepared_batch["perflow_latents_start"]) / (
                    t_end[:, None, None, None] - t_start[:, None, None, None]
                ) * (timepoints[:, None, None, None] - t_start[:, None, None, None])
            
            # Replace the noisy_latents with our interpolated latents for model prediction
            prepared_batch["perflow_latents_t"] = latents_t
            prepared_batch["noisy_latents_original"] = prepared_batch["noisy_latents"].clone()
            prepared_batch["noisy_latents"] = latents_t
            
            # Prepare targets
            if self.config['loss_type'] == "velocity_matching":
                # For velocity matching, the target is the velocity vector
                if len(prepared_batch["latents"].shape) == 5:
                    targets = (latents_end - prepared_batch["perflow_latents_start"]) / (
                        t_end[:, None, None, None, None] - t_start[:, None, None, None, None]
                    )
                else:
                    targets = (latents_end - prepared_batch["perflow_latents_start"]) / (
                        t_end[:, None, None, None] - t_start[:, None, None, None]
                    )
            else:
                raise ValueError(f"Unsupported loss type for flow matching: {self.config['loss_type']}")
                
            prepared_batch["perflow_targets"] = targets
        
        # Log the final state of the batch
        logger.info(f"Final batch keys: {prepared_batch.keys()}")
        if "perflow_targets" in prepared_batch:
            logger.info(f"perflow_targets shape: {prepared_batch['perflow_targets'].shape}")
        
        return prepared_batch
    
    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        """Compute flow matching PeRFlow-specific distillation loss."""
        # For flow matching PeRFlow, we only care about the student model 
        # learning to match the teacher's predictions (velocity/flow field)
        # Get model prediction
        model_pred = model_output["model_prediction"]
        
        # Get targets 
        targets = prepared_batch["perflow_targets"]
        
        # Log shapes for debugging
        logger.info(f"Computing loss - model_pred shape: {model_pred.shape}, targets shape: {targets.shape}")
        
        # Compute velocity matching loss
        loss = F.mse_loss(model_pred.float(), targets.float(), reduction="none")
        
        # Log per-element losses for debugging
        loss_mean = loss.mean().item()
        loss_max = loss.max().item()
        loss_min = loss.min().item()
        logger.info(f"Loss stats - mean: {loss_mean}, max: {loss_max}, min: {loss_min}")
        
        # Reduce the loss
        loss = loss.mean()
        
        logs = {
            "perflow_loss": loss.item(),
            "perflow_loss_max": loss_max,
            "perflow_loss_min": loss_min
        }
        
        return loss, logs
