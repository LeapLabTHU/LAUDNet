from hardware_models.multi_cores import GPGPUDynamicPredictor
import numpy as np
from report import SimulationReport
import math

class PredictorTransformer(GPGPUDynamicPredictor):
    def simulate_unfold(self, in_shape, out_shape):
        all_input = np.prod(in_shape)
        all_output = np.prod(out_shape)
        all_pe_memory = all_output + all_output
        global_latency = (all_input + all_output) / self.mem_fp32_bandwidth
        l2_latency = all_pe_memory / self.l2_fp32_bandwidth
        mem_latency = global_latency + l2_latency
        report = SimulationReport(
            compute_latency=0,
            memory_latency=mem_latency,
            latency=mem_latency+self.launch_time,
        )
        return report
    
    def simulate_matmul(self, a_shape, b_shape, out_shape):
        assert a_shape[-1]==b_shape[-2]
        cin=b_shape[-2]
        cout=b_shape[-1]
        outhw=np.prod(out_shape[:-1])
        outh=round(math.sqrt(outhw))
        outw=round(outhw/outh)
        cfg, best_info, latency = self.search_conv_tile_cfg(
            cout, cin, outh, outw, 1, 1, 1
        )
        if self.verbose:
            # print(f"tile={cfg} efficiency={efficiency}  best_info={best_info} theory compute {compute_conv}")
            print(
                f"tile={cfg} latency={latency}  best_info={best_info}"
            )
        report = SimulationReport(
            compute_latency=best_info["compute_latency"],
            memory_latency=best_info["memory_latency"],
            latency=latency +self.launch_time,
            cfg=cfg,
        )
        return report
    
    def simulate_linear(self, x_shape, w_shape, out_shape):
        b_shape=[_ for _ in w_shape[:-2]]+[w_shape[-1],w_shape[-2]]
        return self.simulate_matmul(x_shape, b_shape, out_shape)
    
    def simulate_elementwise(self, shape):
        if len(shape)>2:
            h=np.prod(shape[:-2])
        else:
            h=1
        w=shape[-2]
        c=shape[-1]
        return self.simulate_add(c,h,w)

    def simulate_reduce(self, shape, reduce_dim):
        n_pixels = 1
        for dim in reduce_dim:
            n_pixels *= shape[dim]
        h=math.ceil(n_pixels**0.5)
        w=math.ceil(n_pixels**0.5)
        c=int(np.prod(shape)/n_pixels)
        best_tile, best_info_pool, latency = self.search_global_avg_pool_tile_cfg(
            c, h, w
        )
        report = SimulationReport(latency=latency+self.launch_time, cfg=best_tile)
        return report
    
    def simulate_softmax(self,shape):
        r_max=self.simulate_reduce(shape,reduce_dim=[-1])
        r_sub_exp=self.simulate_elementwise(shape)
        # r_exp=self.simulate_elementwise(shape)
        r_sum=self.simulate_reduce(shape,reduce_dim=[-1])
        r_div=self.simulate_elementwise(shape)
        report=r_max+r_sub_exp+r_sum+r_div
        report.latency-=self.launch_time*3
        # print(f"DEBUG: r_max={r_max}\n r_sub_exp={r_sub_exp}\n r_sum={r_sum}\n r_div={r_div}\n report={report}")
        return report
    
    def simulate_layernorm(self,shape):
        r_mean=self.simulate_reduce(shape,reduce_dim=[-1])
        # r_sub_var=self.simulate_elementwise(shape)
        r_sub_var=self.simulate_reduce(shape,reduce_dim=[-1])
        r_div=self.simulate_elementwise(shape)
        r_mul_add=self.simulate_elementwise(shape)
        # r_add=self.simulate_elementwise(shape)
        report=r_mean+r_sub_var+r_div+r_mul_add
        report.latency-=self.launch_time*3
        return report
    
    def simulate_gelu(self,shape):
        return self.simulate_elementwise(shape)
    
    def simualte_dylinear(self, x_shape, w_shape, out_shape, ic_density=1,oc_density=1):
        # dyna_data==False and width_select is not None
        b_shape=[_ for _ in w_shape[:-2]]+[w_shape[-1],w_shape[-2]]
        a_shape=[_ for _ in x_shape]
        new_out_shape=[_ for _ in out_shape]
        if ic_density<1:
            a_shape[-1]=round(a_shape[-1]*ic_density)
            b_shape[-2]=round(b_shape[-2]*ic_density)
        if oc_density<1:
            b_shape[-1]=round(b_shape[-1]*oc_density)
            new_out_shape[-1]=round(new_out_shape[-1]*oc_density)
        return self.simulate_matmul(a_shape, b_shape, new_out_shape)